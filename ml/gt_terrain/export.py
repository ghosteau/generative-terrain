"""
Export the trained VAE decoder for the Java plugin.

We export *only the decoder* -- the encoder is a training-time device. The graph
the plugin runs is ``(z, biome_id) -> logits[1, C, X, Y, Z]``. We also write the
two JSON mappings the plugin reads so Python and Java share one source of truth:

* ``block_group_mapping.json``  ``{id: GROUP_NAME}``  (argmax index -> group)
* ``biome_id_mapping.json``     ``{BIOME_NAME: id}``  (must match training order)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import torch.nn as nn

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import BiomeEncoder
from gt_terrain.models import BaselineVoxelNet, ConditionalTerrainVAE, VAEDecoderForExport

ONNX_OPSET = 17


def export_decoder_onnx(
    vae: ConditionalTerrainVAE,
    config: Config,
    path: str | Path | None = None,
) -> Path:
    """Trace the decoder to ONNX. Returns the written path."""
    config.ensure_dirs()
    path = Path(path) if path else config.artifact_dir / "terrain_vae_decoder.onnx"
    shape = (config.chunk_width, config.chunk_height, config.chunk_depth)

    wrapper = VAEDecoderForExport(vae, shape).to("cpu").eval()
    dummy_z = torch.randn(1, config.latent_dim)
    dummy_biome = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_z, dummy_biome),
        str(path),
        input_names=["z", "biome_id"],
        output_names=["logits"],
        dynamic_axes={"z": {0: "batch"}, "biome_id": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET,
    )
    return path


class _BaselineForExport(nn.Module):
    """Wrap the deterministic baseline behind the VAE decoder's I/O contract.

    The plugin always calls the model with ``(z, biome_id)``. The baseline
    ignores ``z`` (it has no latent), but we keep ``z`` as a live graph input --
    via a ``+ 0 * z.sum()`` no-op -- so the exported graph accepts exactly the
    same inputs as the VAE decoder. That makes the baseline a drop-in for an
    immediate in-game test without changing the Java or the mappings.
    """

    def __init__(self, baseline: BaselineVoxelNet, shape: tuple[int, int, int]):
        super().__init__()
        self.baseline = baseline
        self.shape = shape

    def forward(self, z: torch.Tensor, biome_id: torch.Tensor) -> torch.Tensor:
        logits = self.baseline(biome_id, self.shape)
        return logits + 0.0 * z.sum()


def export_baseline_onnx(
    baseline: BaselineVoxelNet,
    config: Config,
    latent_dim: int,
    path: str | Path | None = None,
) -> Path:
    """Export the baseline behind the ``(z, biome_id) -> logits`` interface.

    ``latent_dim`` only sets the (ignored) ``z`` input width so it matches what
    the plugin sends; pass the same value the plugin/VAE use.
    """
    config.ensure_dirs()
    path = Path(path) if path else config.artifact_dir / "terrain_baseline_decoder.onnx"
    shape = (config.chunk_width, config.chunk_height, config.chunk_depth)

    wrapper = _BaselineForExport(baseline, shape).to("cpu").eval()
    dummy_z = torch.randn(1, latent_dim)
    dummy_biome = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        wrapper,
        (dummy_z, dummy_biome),
        str(path),
        input_names=["z", "biome_id"],
        output_names=["logits"],
        dynamic_axes={"z": {0: "batch"}, "biome_id": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET,
    )
    return path


def write_mappings(
    grouping: BlockGrouping,
    biome_encoder: BiomeEncoder,
    config: Config,
) -> tuple[Path, Path]:
    """Write the two JSON files the plugin decodes model output with."""
    config.ensure_dirs()
    group_path = config.artifact_dir / "block_group_mapping.json"
    biome_path = config.artifact_dir / "biome_id_mapping.json"
    grouping.save_group_mapping(group_path)
    biome_encoder.save(biome_path)
    return group_path, biome_path


def verify_onnx(onnx_path: str | Path, config: Config, num_biomes: int) -> tuple:
    """Run the exported graph once with onnxruntime; assert the output shape.

    Mirrors exactly what the Java plugin will do: random ``z`` + a biome id in,
    a ``[1, C, X, Y, Z]`` logit volume out.

    Returns ``None`` (with a warning) if ``onnxruntime`` isn't installed, so the
    notebook's export step never hard-fails just because the optional runtime is
    missing -- the ``.onnx`` is already written by that point.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("[verify_onnx] onnxruntime not installed; skipping verification. "
              "Install it with `pip install onnxruntime` to enable this check.")
        return None

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    z = np.random.randn(1, config.latent_dim).astype(np.float32)
    biome = np.array([np.random.randint(0, num_biomes)], dtype=np.int64)
    out = sess.run(["logits"], {"z": z, "biome_id": biome})[0]

    expected = (1, config.chunk_width, config.chunk_height, config.chunk_depth)
    assert out.shape[0] == 1 and out.shape[2:] == expected[1:], (
        f"Unexpected ONNX output shape {out.shape}"
    )
    return out.shape
