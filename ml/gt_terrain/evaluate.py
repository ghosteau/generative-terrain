"""
Evaluation and plots.

Voxel accuracy is necessary but not sufficient: a model that predicts AIR for the
top half and STONE for the bottom half scores well yet looks nothing like
terrain. So alongside per-group accuracy we measure **terrain plausibility**:

* the *vertical profile* -- for each height Y, the fraction of each block group --
  compared between real and generated chunks. Good terrain has bedrock at the
  floor, a stone body, a thin surface band, then air;
* sample *diversity* -- how much two generated chunks differ, which catches VAE
  posterior collapse (all samples identical).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import IGNORE_INDEX

import matplotlib
matplotlib.use("Agg")            # headless: save files, never pop a window
import matplotlib.pyplot as plt  # noqa: E402


@torch.no_grad()
def baseline_accuracy(model, loader, config: Config) -> dict:
    """Overall + per-group accuracy of the deterministic baseline on a loader."""
    device = torch.device(config.device)
    shape = (config.chunk_width, config.chunk_height, config.chunk_depth)
    model.eval()
    correct = total = 0
    num_classes = model.decoder.head[-1].out_channels
    per_correct = np.zeros(num_classes)
    per_total = np.zeros(num_classes)

    for grid, biome in loader:
        grid, biome = grid.to(device), biome.to(device)
        pred = model(biome, shape).argmax(dim=1)
        mask = grid != IGNORE_INDEX
        correct += (pred[mask] == grid[mask]).sum().item()
        total += mask.sum().item()
        for c in range(num_classes):
            cm = mask & (grid == c)
            per_total[c] += cm.sum().item()
            per_correct[c] += (pred[cm] == c).sum().item()

    per_group = {
        c: (per_correct[c] / per_total[c]) if per_total[c] else float("nan")
        for c in range(num_classes)
    }
    return {"overall": correct / max(total, 1), "per_group": per_group}


def vertical_profile(grids: np.ndarray, num_classes: int) -> np.ndarray:
    """Fraction of each group at each height. Returns ``[Y, num_classes]``.

    ``grids`` is ``[N, X, Y, Z]``. Averaged over N, X, Z.
    """
    y_dim = grids.shape[2]
    prof = np.zeros((y_dim, num_classes))
    for y in range(y_dim):
        layer = grids[:, :, y, :].reshape(-1)
        layer = layer[layer != IGNORE_INDEX]
        if layer.size:
            counts = np.bincount(layer, minlength=num_classes)
            prof[y] = counts / counts.sum()
    return prof


def sample_diversity(grids: np.ndarray) -> float:
    """Mean fraction of voxels that differ between distinct sampled chunks.

    0.0 means all samples are identical (collapse); higher means more variety.
    """
    n = len(grids)
    if n < 2:
        return 0.0
    flat = grids.reshape(n, -1)
    diffs = [
        np.mean(flat[i] != flat[j])
        for i in range(n) for j in range(i + 1, n)
    ]
    return float(np.mean(diffs))


# --- plots -----------------------------------------------------------------
def plot_training_curves(history, path: str | Path, title: str = "Training") -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(history.train, label="train")
    plt.plot(history.val, label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(title)
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130); plt.close()


def plot_vertical_profiles(
    real: np.ndarray,
    generated: np.ndarray,
    grouping: BlockGrouping,
    config: Config,
    path: str | Path,
    top_k: int = 8,
) -> None:
    """Stacked real-vs-generated profile for the most common groups."""
    nc = grouping.num_classes
    rp = vertical_profile(real, nc)
    gp = vertical_profile(generated, nc)
    ys = np.arange(config.chunk_height) + config.min_y

    # Focus on the groups that actually occupy volume.
    top = np.argsort(rp.sum(axis=0))[::-1][:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, prof, name in ((axes[0], rp, "Real"), (axes[1], gp, "Generated")):
        for c in top:
            ax.plot(prof[:, c], ys, label=grouping.idx_to_group[int(c)])
        ax.set_title(f"{name} vertical profile"); ax.set_xlabel("fraction")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("world Y")
    axes[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=130); plt.close()
