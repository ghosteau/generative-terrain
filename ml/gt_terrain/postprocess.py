"""
Post-processing of generated terrain.

The model's job is to get the terrain *shape* right (where ground, surface, air,
and caves are). Two things it does badly are best handled deterministically
afterwards -- the same split Minecraft itself uses (noise terrain + procedural
ore placement):

* :func:`clean_terrain` -- remove single floating blocks and fill single-voxel
  pinholes. This tidies the fuzzy/speckled surface a generative model produces
  near uncertain boundaries, *without* touching real (multi-voxel) caves.
* :func:`scatter_ores` -- sprinkle ore groups into stone/deepslate at realistic
  depths and rarities. Learning rare scattered classes from a VAE is unreliable;
  scattering them procedurally is controllable and always works.

The Java plugin (`TerrainPostProcessor`) mirrors this so in-game generation
matches the notebook.
"""

from __future__ import annotations

import numpy as np

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import IGNORE_INDEX

# Ore -> (min world Y, max world Y, per-eligible-voxel spawn probability).
# Loosely follows vanilla distributions; tune via Config.ore_density.
ORE_SCATTER: list[tuple[str, int, int, float]] = [
    ("COAL_ORE",     0,   256, 0.012),
    ("IRON_ORE",   -64,   256, 0.009),
    ("COPPER_ORE", -16,   112, 0.007),
    ("REDSTONE_ORE", -64,  15, 0.008),
    ("GOLD_ORE",   -64,    32, 0.0030),
    ("LAPIS_ORE",  -64,    64, 0.0025),
    ("DIAMOND_ORE", -64,   16, 0.0018),
    ("EMERALD_ORE", -16,  256, 0.0010),
]


def _neighbor_solid_count(solid: np.ndarray) -> np.ndarray:
    """Count of the 6 face-neighbours that are solid, edge-replicated (no wrap)."""
    p = np.pad(solid.astype(np.int8), 1, mode="edge")
    return (
        p[2:, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]
        + p[1:-1, 2:, 1:-1] + p[1:-1, :-2, 1:-1]
        + p[1:-1, 1:-1, 2:] + p[1:-1, 1:-1, :-2]
    )


def clean_terrain(grid: np.ndarray, grouping: BlockGrouping, config: Config) -> np.ndarray:
    """Despeckle floating blocks and fill single-voxel pinholes. Returns a new grid.

    Only single-voxel anomalies are touched, so multi-voxel caves/overhangs are
    preserved.
    """
    air = grouping.group_to_idx["AIR"]
    stone = grouping.group_to_idx["STONE"]
    deepslate = grouping.group_to_idx["DEEPSLATE"]
    g = grid.copy()

    # 1) Remove solid voxels with zero solid neighbours (floating specks).
    solid = g != air
    g[solid & (_neighbor_solid_count(solid) == 0)] = air

    # 2) Fill air voxels fully enclosed by solid (1-voxel pinholes).
    solid = g != air
    fill = (~solid) & (_neighbor_solid_count(solid) == 6)
    world_y = (np.arange(g.shape[1]) + config.min_y)
    fill_val = np.where(world_y < 0, deepslate, stone)            # [Y]
    fill_val = np.broadcast_to(fill_val[None, :, None], g.shape)
    g[fill] = fill_val[fill]
    return g


def scatter_ores(
    grid: np.ndarray,
    grouping: BlockGrouping,
    config: Config,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Scatter ore groups into stone/deepslate by depth and rarity. Returns new grid."""
    if rng is None:
        rng = np.random.default_rng()
    g = grid.copy()
    stone = grouping.group_to_idx["STONE"]
    deepslate = grouping.group_to_idx["DEEPSLATE"]
    eligible = (g == stone) | (g == deepslate)
    world_y = np.arange(g.shape[1]) + config.min_y

    for name, y_min, y_max, prob in ORE_SCATTER:
        gid = grouping.group_to_idx.get(name)
        if gid is None:
            continue
        band = (world_y >= y_min) & (world_y <= y_max)            # [Y]
        draw = rng.random(g.shape) < (prob * config.ore_density)
        mask = eligible & band[None, :, None] & draw
        g[mask] = gid
        eligible &= ~mask                                         # don't overwrite placed ores
    return g


def postprocess_grid(
    grid: np.ndarray,
    grouping: BlockGrouping,
    config: Config,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Full pipeline: clean (optional) then scatter ores (optional)."""
    g = grid
    # Unlabeled voxels shouldn't appear in generated grids, but guard anyway.
    if (g == IGNORE_INDEX).any():
        g = np.where(g == IGNORE_INDEX, grouping.group_to_idx["AIR"], g)
    if config.clean_terrain:
        g = clean_terrain(g, grouping, config)
    if config.ore_scatter:
        g = scatter_ores(g, grouping, config, rng)
    return g
