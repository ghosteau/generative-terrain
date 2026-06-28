"""
Training loops for both models.

Both share class-weighted cross-entropy with ``ignore_index`` for unlabeled
voxels. The VAE additionally has a KL term with two stabilisers that matter a
lot when data is scarce (we currently have ~230 chunks):

* **KL annealing** -- beta ramps 0 -> target over ``kl_anneal_epochs`` so the
  decoder learns to use ``z`` before the prior pressure kicks in.
* **free bits** -- the first ``free_bits`` nats per latent dim are "free"
  (not penalised), which prevents posterior collapse (the failure mode where the
  VAE ignores ``z`` and every sample looks identical).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import IGNORE_INDEX, ChunkGridDataset, split_indices
from gt_terrain.models import BaselineVoxelNet, ConditionalTerrainVAE

try:                                  # nice progress bars if available, silent fallback
    from tqdm.auto import tqdm
except ImportError:                   # pragma: no cover
    def tqdm(it, **_):
        return it


@dataclass
class History:
    train: list[float]
    val: list[float]
    best_val: float
    best_epoch: int


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(
    grids: np.ndarray, biome_ids: np.ndarray, config: Config
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train/val/test loaders from in-memory grids (augment train only).

    When ``config.balance_biomes`` is set, the training loader draws chunks with a
    biome-balanced sampler (probability ~ 1/biome_count) so the model sees rare
    biomes (RIVER, SAVANNA, ...) about as often as common ones (FOREST, PLAINS)
    instead of overfitting the majority.
    """
    train_idx, val_idx, test_idx = split_indices(len(grids), config)
    common = dict(num_workers=config.num_workers, pin_memory=(config.device == "cuda"))

    train_ds = ChunkGridDataset(grids, biome_ids, train_idx, augment=config.augment)
    if config.balance_biomes and len(train_idx) > 0:
        b = biome_ids[train_idx]
        counts = np.bincount(b, minlength=int(b.max()) + 1)
        weights = 1.0 / np.maximum(counts[b], 1)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), len(train_idx), replacement=True
        )
        train = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler,
                           drop_last=False, **common)
    else:
        train = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                           drop_last=False, **common)
    val = DataLoader(
        ChunkGridDataset(grids, biome_ids, val_idx, augment=False),
        batch_size=config.batch_size, shuffle=False, **common,
    )
    test = DataLoader(
        ChunkGridDataset(grids, biome_ids, test_idx, augment=False),
        batch_size=config.batch_size, shuffle=False, **common,
    )
    return train, val, test


def _voxel_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over every voxel. logits [B,C,X,Y,Z], target [B,X,Y,Z]."""
    c = logits.shape[1]
    logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, c)
    target = target.reshape(-1)
    return F.cross_entropy(logits, target, weight=weight, ignore_index=IGNORE_INDEX)


def _target_heightmap(grid: torch.Tensor, air_id: int) -> torch.Tensor:
    """True per-column surface height (normalised highest non-air local Y) -> [B,1,X,Z]."""
    b, x, y, z = grid.shape
    nonair = (grid != air_id) & (grid != IGNORE_INDEX)
    yidx = torch.arange(y, device=grid.device).view(1, 1, y, 1)
    h = torch.where(nonair, yidx, torch.zeros_like(yidx)).amax(dim=2)   # [B,X,Z]
    return (h.float() / max(y - 1, 1)).unsqueeze(1)


def _kl_with_free_bits(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float) -> torch.Tensor:
    """KL to N(0, I) with free bits, **averaged over latent dims**.

    Averaging over dims (rather than summing) is the critical scale fix: the
    reconstruction term is a per-voxel mean cross-entropy (order ~1), so the KL
    must be on the same per-element scale or it dominates the loss and collapses
    the decoder to a constant (this is exactly what made the first VAE output
    100% air). Averaging also makes ``beta`` interpretable independent of
    ``latent_dim`` (it's equivalent to summing with ``beta/latent_dim``).

    Free bits give each dim ``free_bits`` nats "for free" (no gradient below the
    threshold), which prevents the opposite failure -- posterior collapse, where
    the model ignores ``z`` and every sample looks identical.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # [B, latent]
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    return kl_per_dim.mean()


def _save_best(model: nn.Module, name: str, config: Config) -> None:
    config.ensure_dirs()
    torch.save(model.state_dict(), config.artifact_dir / name)


def train_baseline(
    config: Config,
    grids: np.ndarray,
    biome_ids: np.ndarray,
    class_weights: np.ndarray,
    num_classes: int,
    num_biomes: int,
) -> tuple[BaselineVoxelNet, History]:
    """Train the deterministic baseline; saves ``baseline.pth``."""
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, _ = make_loaders(grids, biome_ids, config)

    model = BaselineVoxelNet(num_classes, num_biomes, config).to(device)
    weight = torch.tensor(class_weights, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    hist = History([], [], float("inf"), -1)
    patience = 0
    shape = (config.chunk_width, config.chunk_height, config.chunk_depth)

    for epoch in range(config.epochs):
        model.train()
        tl = 0.0
        for grid, biome in tqdm(train_loader, desc=f"baseline e{epoch+1} train", leave=False):
            grid, biome = grid.to(device), biome.to(device)
            opt.zero_grad()
            logits = model(biome, shape)
            loss = _voxel_ce(logits, grid, weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            tl += loss.item()
        hist.train.append(tl / max(len(train_loader), 1))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for grid, biome in val_loader:
                grid, biome = grid.to(device), biome.to(device)
                vl += _voxel_ce(model(biome, shape), grid, weight).item()
        hist.val.append(vl / max(len(val_loader), 1))
        print(f"[baseline] epoch {epoch+1} train {hist.train[-1]:.4f} val {hist.val[-1]:.4f}")

        if hist.val[-1] < hist.best_val:
            hist.best_val, hist.best_epoch, patience = hist.val[-1], epoch, 0
            _save_best(model, "baseline.pth", config)
        else:
            patience += 1
            if patience >= config.patience:
                print(f"[baseline] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(config.artifact_dir / "baseline.pth", map_location=device))
    return model, hist


def train_vae(
    config: Config,
    grids: np.ndarray,
    biome_ids: np.ndarray,
    class_weights: np.ndarray,
    num_classes: int,
    num_biomes: int,
) -> tuple[ConditionalTerrainVAE, History]:
    """Train the conditional VAE; saves ``vae.pth``."""
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, _ = make_loaders(grids, biome_ids, config)

    model = ConditionalTerrainVAE(num_classes, num_biomes, config).to(device)
    weight = torch.tensor(class_weights, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    air_id = BlockGrouping().group_to_idx["AIR"]
    hw = config.heightmap_weight

    hist = History([], [], float("inf"), -1)
    patience = 0

    for epoch in range(config.epochs):
        beta = config.beta * min(1.0, (epoch + 1) / max(config.kl_anneal_epochs, 1))
        model.train()
        tl = 0.0
        for grid, biome in tqdm(train_loader, desc=f"vae e{epoch+1} train", leave=False):
            grid, biome = grid.to(device), biome.to(device)
            opt.zero_grad()
            logits, mu, logvar, height = model(grid, biome)
            recon = _voxel_ce(logits, grid, weight)
            kl = _kl_with_free_bits(mu, logvar, config.free_bits)
            hloss = F.l1_loss(height, _target_heightmap(grid, air_id))
            loss = recon + beta * kl + hw * hloss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            tl += loss.item()
        hist.train.append(tl / max(len(train_loader), 1))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for grid, biome in val_loader:
                grid, biome = grid.to(device), biome.to(device)
                logits, mu, logvar, height = model(grid, biome)
                recon = _voxel_ce(logits, grid, weight)
                kl = _kl_with_free_bits(mu, logvar, config.free_bits)
                hloss = F.l1_loss(height, _target_heightmap(grid, air_id))
                vl += (recon + beta * kl + hw * hloss).item()
        hist.val.append(vl / max(len(val_loader), 1))
        print(f"[vae] epoch {epoch+1} beta {beta:.3f} train {hist.train[-1]:.4f} val {hist.val[-1]:.4f}")

        if hist.val[-1] < hist.best_val:
            hist.best_val, hist.best_epoch, patience = hist.val[-1], epoch, 0
            _save_best(model, "vae.pth", config)
        else:
            patience += 1
            if patience >= config.patience:
                print(f"[vae] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(config.artifact_dir / "vae.pth", map_location=device))
    return model, hist
