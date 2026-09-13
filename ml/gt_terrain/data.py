"""
Data pipeline: raw CSV chunk exports -> voxel grids the models train on.

Each CSV (produced in-game by ``/grabchunkdata``) is one full chunk:
16 x 384 x 16 = 98,304 rows, one per voxel, with columns::

    x, y, z, ChunkBiome, Biome, Block_ID, Is_Surface, Light_Level,
    Block_to_Left, Block_to_Right, Block_Below, Block_Above,
    Block_in_Front, Block_Behind

We deliberately use only three of them:

* ``Block_ID``    -> the prediction *target* (mapped to a 27-way group id);
* ``ChunkBiome``  -> a per-chunk conditioning label (one biome per chunk);
* ``x, y, z``     -> where each target voxel sits in the grid.

Everything else (the six neighbour columns, ``Is_Surface``, ``Light_Level``) is
ignored on purpose: those describe terrain that does not exist yet at generation
time, so feeding them to the model is leakage. See the package docstring.

The whole dataset is small once gridded (~100 KB of int8 per chunk), so we load
it entirely into RAM as one array instead of writing thousands of per-chunk
``.pt`` files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config

# Sentinel target id for voxels with no label (used as CrossEntropy ignore_index).
# Full chunks cover every voxel, so this only matters for partial/corrupt exports.
IGNORE_INDEX = -1

# Columns we actually read. (Listing them documents intent and speeds up pandas.)
_USED_COLUMNS = ["x", "y", "z", "ChunkBiome", "Block_ID"]


# Reserved biome label. It is always present in the vocabulary, and any biome
# name the encoder has never seen maps to it. During base training a small
# fraction of chunks is relabelled to UNKNOWN (config.biome_dropout) so its
# embedding row learns "generic terrain" -- that way fine-tuning data from
# custom worlds with unrecognised biomes still conditions on something trained.
UNKNOWN_BIOME = "UNKNOWN"


class BiomeEncoder:
    """Deterministic biome-name <-> id mapping (sorted for stability)."""

    def __init__(self, names: list[str]):
        self.names = sorted(set(names) | {UNKNOWN_BIOME})
        self.name_to_id = {n: i for i, n in enumerate(self.names)}

    @property
    def num_biomes(self) -> int:
        return len(self.names)

    @property
    def unknown_id(self) -> int:
        return self.name_to_id[UNKNOWN_BIOME]

    def transform(self, name: str) -> int:
        return self.name_to_id.get(name, self.unknown_id)

    def save(self, path: str | Path) -> None:
        """Write ``{BIOME_NAME: id}`` for the Java plugin to load."""
        Path(path).write_text(json.dumps(self.name_to_id, indent=2), encoding="utf-8")

    @classmethod
    def from_csvs(cls, csv_files: list[Path]) -> "BiomeEncoder":
        names: set[str] = set()
        for f in csv_files:
            # ChunkBiome is one value for the whole chunk, so a 1-row read is enough.
            names.add(str(pd.read_csv(f, usecols=["ChunkBiome"], nrows=1)["ChunkBiome"][0]))
        return cls(sorted(names))

    @classmethod
    def from_mapping(cls, name_to_id: dict[str, int]) -> "BiomeEncoder":
        """Rebuild from a saved ``{BIOME_NAME: id}`` mapping, preserving ids exactly.

        Used by fine-tuning: the ids must match the embedding rows of the base
        checkpoint, so we restore the mapping verbatim instead of re-deriving it.
        """
        enc = cls.__new__(cls)
        enc.names = [n for n, _ in sorted(name_to_id.items(), key=lambda kv: kv[1])]
        enc.name_to_id = dict(name_to_id)
        if UNKNOWN_BIOME not in enc.name_to_id:
            # Pre-style-era mapping: tolerate it, fall back to id 0 like before.
            enc.name_to_id[UNKNOWN_BIOME] = 0
        return enc

    @classmethod
    def load(cls, path: str | Path) -> "BiomeEncoder":
        return cls.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def list_csv_files(config: Config) -> list[Path]:
    """All chunk CSVs under ``config.data_dir`` (sorted, optionally capped)."""
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"Set GT_DATA_DIR or export chunks with /grabchunkdata first."
        )
    files = sorted(data_dir.glob("*.csv"))
    if config.max_chunks is not None:
        files = files[: config.max_chunks]
    if not files:
        raise FileNotFoundError(f"No .csv chunk files found in {data_dir}")
    return files


def csv_to_grid(
    csv_path: Path,
    grouping: BlockGrouping,
    config: Config,
) -> np.ndarray:
    """Convert one chunk CSV to a target group grid of shape [X, Y, Z].

    Vectorised end to end: read the needed columns, map block names to group ids
    in one pass, then scatter into the grid by integer indexing. No Python-level
    per-row loop and no ``inverse_transform`` (the old bottleneck).
    """
    df = pd.read_csv(csv_path, usecols=_USED_COLUMNS)

    x = df["x"].to_numpy(np.int64)
    y = df["y"].to_numpy(np.int64)
    z = df["z"].to_numpy(np.int64)
    local_y = y - config.min_y  # world Y (-64..319) -> grid Y (0..383)

    # Keep only in-bounds voxels (guards against stray rows).
    in_bounds = (
        (x >= 0) & (x < config.chunk_width)
        & (local_y >= 0) & (local_y < config.chunk_height)
        & (z >= 0) & (z < config.chunk_depth)
    )

    group_ids = grouping.map_block_array(df["Block_ID"].to_numpy(dtype=object))

    grid = np.full(
        (config.chunk_width, config.chunk_height, config.chunk_depth),
        IGNORE_INDEX,
        dtype=np.int8,          # 27 classes + the -1 sentinel fit comfortably
    )
    grid[x[in_bounds], local_y[in_bounds], z[in_bounds]] = group_ids[in_bounds]
    return grid


def build_dataset(
    config: Config,
    grouping: BlockGrouping,
    biome_encoder: BiomeEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Grid every CSV. Returns ``(grids [N,X,Y,Z] int8, biome_ids [N] int64)``."""
    files = list_csv_files(config)
    grids = np.empty(
        (len(files), config.chunk_width, config.chunk_height, config.chunk_depth),
        dtype=np.int8,
    )
    biome_ids = np.empty(len(files), dtype=np.int64)

    for i, f in enumerate(files):
        grids[i] = csv_to_grid(f, grouping, config)
        # One biome per chunk: read the first ChunkBiome value.
        biome_name = str(pd.read_csv(f, usecols=["ChunkBiome"], nrows=1)["ChunkBiome"][0])
        biome_ids[i] = biome_encoder.transform(biome_name)

    return grids, biome_ids


def compute_class_weights(grids: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-sqrt-frequency class weights, clipped and mean-normalised.

    Terrain is dominated by AIR and STONE; without weighting the model just
    predicts those everywhere. Same recipe as the historical pipeline, kept
    because it was one of the few sound parts of it.
    """
    flat = grids.reshape(-1)
    flat = flat[flat != IGNORE_INDEX]
    counts = np.bincount(flat, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)            # avoid divide-by-zero
    freq = counts / counts.sum()
    weights = (1.0 / freq) ** 0.5
    weights = np.clip(weights, 0.1, 5.0)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


def split_indices(n: int, config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle [0..n) with the configured seed and split train/val/test."""
    rng = np.random.default_rng(config.seed)
    idx = rng.permutation(n)
    n_val = int(n * config.val_split)
    n_test = int(n * config.test_split)
    test = idx[:n_test]
    val = idx[n_test : n_test + n_val]
    train = idx[n_test + n_val :]
    return train, val, test


class ChunkGridDataset(Dataset):
    """Serves ``(target_grid [X,Y,Z] long, biome_id long)`` pairs.

    Augmentation (training only) applies the 4 horizontal rotations and the two
    horizontal mirror flips. Crucially it never flips the Y axis -- terrain has a
    hard up/down asymmetry (sky on top, bedrock at the bottom), so a vertical
    flip would create physically impossible training examples.
    """

    def __init__(
        self,
        grids: np.ndarray,
        biome_ids: np.ndarray,
        indices: np.ndarray,
        augment: bool = False,
    ):
        self.grids = grids
        self.biome_ids = biome_ids
        self.indices = indices
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        grid = torch.from_numpy(self.grids[idx].astype(np.int64))  # [X,Y,Z]
        biome = torch.tensor(int(self.biome_ids[idx]), dtype=torch.long)

        if self.augment:
            k = int(torch.randint(0, 4, (1,)).item())        # horizontal rotation
            if k:
                grid = torch.rot90(grid, k, dims=(0, 2))
            if torch.rand(1).item() < 0.5:                   # mirror along X
                grid = torch.flip(grid, dims=(0,))
            if torch.rand(1).item() < 0.5:                   # mirror along Z
                grid = torch.flip(grid, dims=(2,))

        return grid.contiguous(), biome
