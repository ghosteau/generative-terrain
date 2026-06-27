"""
Runtime configuration.

Everything that used to be a magic constant scattered across notebook cells
(paths, chunk geometry, hyper-parameters) lives here in one immutable dataclass.
Paths can be overridden with environment variables so the same code runs on the
training box (GPU) and on a CI / smoke-test machine (CPU) without edits:

    GT_DATA_DIR      directory of raw ``*.csv`` chunk exports
    GT_ARTIFACT_DIR  directory for models, mappings, and plots

A ``tiny()`` preset is provided for fast CPU smoke tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

# --- Minecraft world geometry (1.21). These are fixed by the game, not tunable.
CHUNK_WIDTH = 16          # blocks along X
CHUNK_DEPTH = 16          # blocks along Z
MIN_Y = -64               # world floor
MAX_Y = 319               # world ceiling
CHUNK_HEIGHT = MAX_Y - MIN_Y + 1  # 384 blocks along Y


def _default_data_dir() -> Path:
    """Where raw CSV chunk exports live.

    Defaults to the PaperMC server export folder used by the plugin's
    ``/grabchunkdata`` command, overridable via ``GT_DATA_DIR``.
    """
    env = os.environ.get("GT_DATA_DIR")
    if env:
        return Path(env)
    return Path(r"C:\Users\cathe\Desktop\Plugins\Server\block_dataset")


def _default_artifact_dir() -> Path:
    env = os.environ.get("GT_ARTIFACT_DIR")
    if env:
        return Path(env)
    # Sibling of the package: ml/artifacts
    return Path(__file__).resolve().parent.parent / "artifacts"


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class Config:
    """Immutable bundle of paths, geometry, and hyper-parameters.

    Use :meth:`with_` to derive a modified copy (it is frozen on purpose so a
    config can be logged / hashed and trusted not to change mid-run).
    """

    # Paths
    data_dir: Path = field(default_factory=_default_data_dir)
    artifact_dir: Path = field(default_factory=_default_artifact_dir)

    # Geometry (mirrors the constants above; kept on the config for convenience)
    chunk_width: int = CHUNK_WIDTH
    chunk_depth: int = CHUNK_DEPTH
    chunk_height: int = CHUNK_HEIGHT
    min_y: int = MIN_Y
    max_y: int = MAX_Y

    # Data
    max_chunks: int | None = None     # cap files loaded (None = all); set small for smoke tests
    val_split: float = 0.15
    test_split: float = 0.15
    augment: bool = True              # rotate/flip augmentation in the dataset

    # Shared model dims
    biome_embed_dim: int = 16
    base_channels: int = 32           # width of the 3D conv stacks

    # VAE
    latent_dim: int = 64
    # KL is averaged per latent dim (see train._kl_with_free_bits), so it's on the
    # same scale as the per-voxel reconstruction loss and beta ~1 is balanced.
    beta: float = 1.0                 # KL weight (target after annealing)
    kl_anneal_epochs: int = 10        # epochs to ramp beta 0 -> beta
    free_bits: float = 0.02           # nats per latent dim that incur no KL penalty

    # Optimisation
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20                # early-stopping patience (epochs)
    grad_clip: float = 1.0
    num_workers: int = 0
    seed: int = 1337

    device: str = field(default_factory=_pick_device)

    # --- convenience -------------------------------------------------------
    def with_(self, **overrides) -> "Config":
        """Return a copy with the given fields replaced."""
        return replace(self, **overrides)

    @property
    def voxels_per_chunk(self) -> int:
        return self.chunk_width * self.chunk_height * self.chunk_depth

    def ensure_dirs(self) -> None:
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tiny() -> "Config":
        """A fast CPU preset for tests and smoke runs (not for real training)."""
        return Config(
            max_chunks=8,
            base_channels=8,
            biome_embed_dim=4,
            latent_dim=8,
            batch_size=2,
            epochs=2,
            patience=2,
            kl_anneal_epochs=1,
            num_workers=0,
            device="cpu",
            augment=False,
        )


def default_config() -> Config:
    """The standard configuration (GPU if available)."""
    return Config()
