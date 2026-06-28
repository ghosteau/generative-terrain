"""
Generation: sample fresh chunks from a trained VAE.

This is the payoff of the redesign -- terrain produced from noise + biome alone,
with no leaked neighbour information. The Java plugin does the same thing at
runtime via the exported ONNX decoder; this module is the Python mirror used for
evaluation and for producing a CSV you can eyeball.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.models import ConditionalTerrainVAE
from gt_terrain.postprocess import postprocess_grid


@torch.no_grad()
def sample_grids(
    vae: ConditionalTerrainVAE,
    biome_id: int,
    n: int,
    config: Config,
    temperature: float = 1.0,
    postprocess: bool = True,
    grouping: BlockGrouping | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``n`` chunks for one biome. Returns group-id grids ``[n,X,Y,Z]``.

    ``temperature`` scales the prior std: 1.0 matches training; <1 yields more
    typical (smoother) terrain, >1 more varied/risky terrain.

    With ``postprocess`` (default), each sample is cleaned (despeckle / fill
    pinholes) and has ores scattered in -- the same steps the plugin applies in
    game. Pass ``postprocess=False`` to inspect the raw model output.
    """
    vae.eval()
    device = torch.device(config.device)
    shape = (config.chunk_width, config.chunk_height, config.chunk_depth)

    z = torch.randn(n, *config.vae_latent_shape, device=device) * temperature
    biome = torch.full((n,), int(biome_id), dtype=torch.long, device=device)
    logits = vae.decode(z, biome, shape)              # [n, C, X, Y, Z]
    preds = logits.argmax(dim=1).cpu().numpy().astype(np.int16)

    if postprocess:
        grouping = grouping or BlockGrouping()
        rng = rng or np.random.default_rng()
        preds = np.stack([postprocess_grid(preds[i], grouping, config, rng) for i in range(n)])
    return preds


def group_to_block(group_name: str, world_y: int) -> str:
    """Pick a representative concrete block for a group (height-aware).

    Mirrors the plugin's ``expandGroupToBlock`` for the cases that depend on
    height only, so the CSV preview looks like what the plugin would place.
    Context-dependent cases (logs vs leaves) just use a sensible default here.
    """
    if group_name.endswith("_ORE"):
        return f"DEEPSLATE_{group_name}" if world_y < 0 else group_name
    if group_name == "DEEPSLATE":
        return "DEEPSLATE" if world_y < 0 else "STONE"
    if group_name.endswith("_WOOD"):
        return group_name.replace("_WOOD", "_LOG")
    if group_name == "GRASS":
        return "GRASS_BLOCK"
    if group_name == "VEGETATION":
        return "SHORT_GRASS"
    if group_name == "MISC":
        return "STONE"
    return group_name  # AIR, STONE, DIRT, SAND, WATER, ... are already block names


def grid_to_csv(
    grid: np.ndarray,
    grouping: BlockGrouping,
    config: Config,
    path: str | Path,
) -> pd.DataFrame:
    """Write one group-id grid ``[X,Y,Z]`` to a block-name CSV (x, y, z, Block_ID)."""
    xs, ys, zs = np.indices(grid.shape)
    world_y = ys.reshape(-1) + config.min_y
    gids = grid.reshape(-1)

    names = np.array([grouping.idx_to_group.get(int(g), "AIR") for g in gids])
    blocks = np.array([group_to_block(n, int(wy)) for n, wy in zip(names, world_y)])

    df = pd.DataFrame(
        {
            "x": xs.reshape(-1),
            "y": ys.reshape(-1),          # local y (0..383); add min_y for world y
            "z": zs.reshape(-1),
            "Block_ID": blocks,
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
