"""
Tests for the data pipeline -- the part where the old project went wrong.

The headline test is :func:`test_no_leakage`: it asserts that the columns we
identified as leakage (the six neighbour blocks, ``Is_Surface``, ``Light_Level``)
are never read by the grid builder. If someone "helpfully" adds them back as
inputs, this fails loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gt_terrain import data
from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import (
    IGNORE_INDEX,
    BiomeEncoder,
    ChunkGridDataset,
    csv_to_grid,
)

LEAKAGE_COLUMNS = {
    "Block_to_Left", "Block_to_Right", "Block_Below", "Block_Above",
    "Block_in_Front", "Block_Behind", "Is_Surface", "Light_Level", "Biome",
}


def _write_fake_chunk(path, config: Config) -> None:
    """A tiny but complete chunk CSV with the full real column set."""
    rows = []
    for x in range(config.chunk_width):
        for y in range(config.min_y, config.max_y + 1):
            for z in range(config.chunk_depth):
                # bedrock floor, stone body, grass at y=0, air above -> plausible
                if y == config.min_y:
                    block = "BEDROCK"
                elif y < 0:
                    block = "STONE"
                elif y == 0:
                    block = "GRASS_BLOCK"
                else:
                    block = "AIR"
                rows.append({
                    "x": x, "y": y, "z": z,
                    "ChunkBiome": "PLAINS", "Biome": "PLAINS",
                    "Block_ID": block, "Is_Surface": False, "Light_Level": 0.0,
                    "Block_to_Left": "STONE", "Block_to_Right": "STONE",
                    "Block_Below": "STONE", "Block_Above": "AIR",
                    "Block_in_Front": "STONE", "Block_Behind": "STONE",
                })
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def tiny(tmp_path):
    cfg = Config.tiny().with_(data_dir=tmp_path, artifact_dir=tmp_path / "art")
    _write_fake_chunk(tmp_path / "plains1.csv", cfg)
    return cfg


def test_no_leakage(monkeypatch, tiny):
    """The grid builder must read ONLY x,y,z,ChunkBiome,Block_ID."""
    real_read_csv = pd.read_csv
    seen_columns: set[str] = set()

    def spy(*args, **kwargs):
        cols = kwargs.get("usecols")
        if cols:
            seen_columns.update(cols)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    grouping = BlockGrouping()
    csv_to_grid(tiny.data_dir / "plains1.csv", grouping, tiny)

    assert seen_columns & LEAKAGE_COLUMNS == set(), (
        f"Leakage columns were read by the builder: {seen_columns & LEAKAGE_COLUMNS}"
    )


def test_grid_shape_and_dtype(tiny):
    grouping = BlockGrouping()
    grid = csv_to_grid(tiny.data_dir / "plains1.csv", grouping, tiny)
    assert grid.shape == (tiny.chunk_width, tiny.chunk_height, tiny.chunk_depth)
    assert grid.dtype == np.int16
    # Full chunk => no ignore voxels remain.
    assert (grid == IGNORE_INDEX).sum() == 0


def test_grid_content_matches_blocks(tiny):
    grouping = BlockGrouping()
    grid = csv_to_grid(tiny.data_dir / "plains1.csv", grouping, tiny)
    bedrock = grouping.group_to_idx["BEDROCK"]
    grass = grouping.group_to_idx["GRASS"]
    air = grouping.group_to_idx["AIR"]
    assert grid[0, 0, 0] == bedrock                  # local y 0 == world y -64
    assert grid[0, 0 - tiny.min_y, 0] == grass       # world y 0
    assert grid[0, tiny.chunk_height - 1, 0] == air  # top


def test_build_and_class_weights(tiny):
    grouping = BlockGrouping()
    enc = BiomeEncoder.from_csvs([tiny.data_dir / "plains1.csv"])
    grids, biome_ids = data.build_dataset(tiny, grouping, enc)
    assert grids.shape[0] == 1
    assert biome_ids.tolist() == [enc.transform("PLAINS")]
    w = data.compute_class_weights(grids, grouping.num_classes)
    assert w.shape == (grouping.num_classes,)
    assert np.isfinite(w).all() and (w > 0).all()


def test_augmentation_preserves_shape_and_labels(tiny):
    grouping = BlockGrouping()
    grids = np.stack([csv_to_grid(tiny.data_dir / "plains1.csv", grouping, tiny)])
    biome_ids = np.array([0])
    ds = ChunkGridDataset(grids, biome_ids, np.array([0]), augment=True)
    grid, biome = ds[0]
    assert grid.shape == (tiny.chunk_width, tiny.chunk_height, tiny.chunk_depth)
    # Horizontal aug must not move the bedrock floor off the bottom layer.
    bedrock = grouping.group_to_idx["BEDROCK"]
    assert (grid[:, 0, :] == bedrock).all()
