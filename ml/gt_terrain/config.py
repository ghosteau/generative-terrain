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
    balance_biomes: bool = True       # weighted sampling so rare biomes aren't drowned out

    # Shared model dims
    biome_embed_dim: int = 16
    base_channels: int = 32           # width of the 3D conv stacks

    # VAE -- the latent is a small SPATIAL grid, not a single vector. A vector
    # latent broadcast uniformly can only produce the average column (flat
    # terrain); a coarse 3D latent lets different regions of the chunk differ,
    # which is what yields hills, valleys, and caves.
    latent_channels: int = 8          # channels per latent grid cell
    # (x, y, z) latent resolution. The Y axis is kept relatively fine so the model
    # can localise where the surface sits per column (coarse Y -> fuzzy/carved
    # surfaces); 4x4 horizontal cells give hill/valley variation.
    latent_grid: tuple = (4, 16, 4)
    latent_dim: int = 64              # width of the (ignored) z for baseline ONNX export
    # KL is averaged per latent element (see train._kl_with_free_bits), so it's on
    # the same scale as the per-voxel reconstruction loss and beta ~1 is balanced.
    beta: float = 1.0                 # KL weight (target after annealing)
    kl_anneal_epochs: int = 10        # epochs to ramp beta 0 -> beta
    free_bits: float = 0.02           # nats per latent element that incur no KL penalty
    # The decoder predicts a per-column surface heightmap internally and feeds a
    # signed-distance-to-surface channel to the voxel head, so it commits to a
    # surface instead of hedging (the "digs out air" problem). This weights the
    # heightmap supervision term during training.
    heightmap_weight: float = 1.0

    # --- Styles (the fine-tuning / transfer-learning mechanism) -------------
    # The decoder is conditioned on a style embedding via FiLM (feature-wise
    # scale/shift at every U-Net block and in the heightmap head). Style 0 is
    # the vanilla base style; the remaining slots are reserved for custom
    # styles learned later by fine-tuning on a user's own chunks -- without
    # retraining the base. One exported model can hold several styles.
    style_dim: int = 64               # width of a style vector
    max_custom_styles: int = 8        # reserved trainable custom-style slots
    # During base training, Gaussian noise of this std is added to the style
    # vector so the FiLM response is smooth around the base style. That makes
    # the style space well-conditioned for fine-tuning (a new style vector
    # moves through terrain that degrades gracefully, not chaotically).
    style_noise: float = 0.1
    # With this probability a training chunk's biome id is replaced by the
    # UNKNOWN biome, so the UNKNOWN row learns "generic terrain". Fine-tuning
    # data whose biomes the base never saw maps to UNKNOWN and still works.
    biome_dropout: float = 0.05

    # --- Fine-tuning ---------------------------------------------------------
    ft_lr: float = 1e-3               # few trainable params -> higher LR is fine
    ft_epochs: int = 60
    ft_patience: int = 15
    # What to unfreeze beyond the new style row:
    #   "style"   -- only the style embedding row (tiny; needs very similar terrain)
    #   "film"    -- + the FiLM projections (default; adapter-sized, recommended)
    #   "decoder" -- + the whole decoder & heightmap head (for 100+ chunk datasets)
    ft_unfreeze: str = "film"

    # Versioned-base bookkeeping: base checkpoints are written to
    # artifact_dir/base_<base_version>/ with a manifest that fine_tune() uses
    # to rebuild the exact architecture and mappings.
    base_version: str = "v1"

    # Post-processing of generated terrain (model predicts shape; we clean + scatter ores).
    clean_terrain: bool = True        # remove floating specks / fill 1-voxel pinholes (keeps caves)
    ore_scatter: bool = True          # procedurally scatter ores into stone/deepslate
    ore_density: float = 1.0          # global multiplier on ore spawn probabilities

    # Optimisation
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20                # early-stopping patience (epochs)
    grad_clip: float = 1.0
    warmup_epochs: int = 5            # linear LR warmup, then cosine decay to ~0
    amp: bool = True                  # mixed-precision autocast on CUDA (ignored on CPU)
    ema_decay: float = 0.999          # EMA of weights; EMA weights are validated/exported
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

    @property
    def vae_latent_shape(self) -> tuple:
        """Full latent tensor shape per sample: (channels, x, y, z)."""
        return (self.latent_channels, *self.latent_grid)

    def ensure_dirs(self) -> None:
        Path(self.artifact_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tiny() -> "Config":
        """A fast CPU preset for tests and smoke runs (not for real training)."""
        return Config(
            max_chunks=8,
            base_channels=8,
            biome_embed_dim=4,
            latent_channels=4,
            latent_grid=(2, 4, 2),
            latent_dim=8,
            style_dim=8,
            max_custom_styles=2,
            batch_size=2,
            epochs=2,
            patience=2,
            kl_anneal_epochs=1,
            warmup_epochs=1,
            ft_epochs=2,
            ft_patience=2,
            num_workers=0,
            device="cpu",
            augment=False,
        )


def default_config() -> Config:
    """The standard configuration (GPU if available)."""
    return Config()
