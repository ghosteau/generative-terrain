"""
Tests for the style system and the fine-tuning (transfer-learning) path.

The end-to-end test is the important one: it trains a tiny base, freezes it
into a versioned checkpoint, fine-tunes a new style from the manifest alone,
and then asserts the two guarantees the design makes:

* the frozen weights (conv kernels, base style row) are bit-identical after
  fine-tuning -- the base style cannot be damaged by a fine-tune;
* the fine-tuned model still exports through the normal ONNX path with the
  ``style_id`` input the plugin feeds.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import (
    UNKNOWN_BIOME,
    BiomeEncoder,
    build_dataset,
    compute_class_weights,
)
from gt_terrain.export import export_decoder_onnx, verify_onnx, write_mappings
from gt_terrain.models import ConditionalTerrainVAE
from gt_terrain.train import fine_tune, save_base_checkpoint, train_vae

from test_data import _write_fake_chunk


@pytest.fixture(scope="module")
def tiny_env(tmp_path_factory):
    """One fake chunk + a tiny config, shared by the tests in this module."""
    tmp = tmp_path_factory.mktemp("ft")
    cfg = Config.tiny().with_(data_dir=tmp, artifact_dir=tmp / "art")
    _write_fake_chunk(tmp / "plains1.csv", cfg)
    return cfg


def test_biome_encoder_reserves_unknown():
    enc = BiomeEncoder(["PLAINS", "FOREST"])
    assert UNKNOWN_BIOME in enc.names
    assert enc.num_biomes == 3
    # Anything unseen maps to the UNKNOWN row, not to an arbitrary real biome.
    assert enc.transform("SOME_MODDED_BIOME") == enc.unknown_id
    assert enc.transform("PLAINS") != enc.unknown_id


def test_biome_encoder_mapping_round_trip(tmp_path):
    enc = BiomeEncoder(["PLAINS", "TAIGA"])
    enc.save(tmp_path / "biomes.json")
    loaded = BiomeEncoder.load(tmp_path / "biomes.json")
    assert loaded.name_to_id == enc.name_to_id     # ids preserved verbatim
    assert loaded.unknown_id == enc.unknown_id


def test_style_conditioned_decode_shapes(tiny_env):
    cfg = tiny_env
    model = ConditionalTerrainVAE(num_classes=27, num_biomes=3, config=cfg).eval()
    shape = (cfg.chunk_width, cfg.chunk_height, cfg.chunk_depth)
    z = torch.randn(2, *cfg.vae_latent_shape)
    biome = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        base = model.decode(z, biome, shape)                       # default style 0
        styled = model.decode(
            z, biome, shape, style_id=torch.ones(2, dtype=torch.long)
        )
    assert base.shape == (2, 27, *shape)
    assert styled.shape == (2, 27, *shape)
    # Untrained custom style rows are zero-initialised, like the base row, so
    # before any fine-tuning the styles are interchangeable (inert FiLM).
    assert torch.allclose(base, styled)


def test_finetune_end_to_end(tiny_env):
    cfg = tiny_env
    grouping = BlockGrouping()
    enc = BiomeEncoder.from_csvs([cfg.data_dir / "plains1.csv"])
    grids, biome_ids = build_dataset(cfg, grouping, enc)
    weights = compute_class_weights(grids, grouping.num_classes)

    vae, _ = train_vae(
        cfg, grids, biome_ids, weights, grouping.num_classes, enc.num_biomes,
        unknown_biome_id=enc.unknown_id,
    )
    base_dir = save_base_checkpoint(vae, cfg, grouping, enc, dataset_chunks=len(grids))
    assert (base_dir / "vae_base.pth").exists()
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["arch"]["style_dim"] == cfg.style_dim

    base_state = {k: v.clone() for k, v in vae.state_dict().items()}

    res = fine_tune(base_dir, cfg.data_dir, "myworld", config=cfg)
    assert res.style_id == 1
    assert res.styles == {"BASE": 0, "myworld": 1}

    ft_state = res.model.state_dict()
    # Frozen guarantees ("film" unfreeze): conv kernels and the BASE style row
    # are bit-identical; the new style row and FiLM projections moved.
    assert torch.equal(
        ft_state["decoder.in_conv.conv1.weight"], base_state["decoder.in_conv.conv1.weight"]
    )
    assert torch.equal(ft_state["encoder.to_latent.weight"], base_state["encoder.to_latent.weight"])
    assert torch.equal(ft_state["style_embed.weight"][0], base_state["style_embed.weight"][0])
    changed = not torch.equal(ft_state["style_embed.weight"][1], base_state["style_embed.weight"][1])
    film_changed = not torch.equal(
        ft_state["decoder.in_conv.film.proj.weight"],
        base_state["decoder.in_conv.film.proj.weight"],
    )
    assert changed or film_changed, "fine-tune trained nothing"

    # Registry ships next to the ONNX for the plugin.
    styles_on_disk = json.loads(
        (cfg.artifact_dir / "style_mapping.json").read_text(encoding="utf-8")
    )
    assert styles_on_disk == {"BASE": 0, "myworld": 1}

    # --- export with the style input, verified the way the plugin runs it ----
    onnx_path = export_decoder_onnx(res.model, res.config)
    write_mappings(res.grouping, res.biome_encoder, res.config, styles=res.styles)
    shape = verify_onnx(onnx_path, res.config, num_biomes=res.biome_encoder.num_biomes)
    if shape is not None:            # onnxruntime installed -> actually ran
        assert shape == (1, 27, cfg.chunk_width, cfg.chunk_height, cfg.chunk_depth)

    import onnx  # required by torch.onnx.export, so present here
    graph_inputs = [i.name for i in onnx.load(str(onnx_path)).graph.input]
    assert graph_inputs == ["z", "biome_id", "style_id"]
