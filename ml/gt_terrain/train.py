"""
Training loops, the versioned base checkpoint, and fine-tuning.

Both models share class-weighted cross-entropy with ``ignore_index`` for
unlabeled voxels. The VAE additionally has a KL term with two stabilisers that
matter a lot when data is scarce:

* **KL annealing** -- beta ramps 0 -> target over ``kl_anneal_epochs`` so the
  decoder learns to use ``z`` before the prior pressure kicks in.
* **free bits** -- the first ``free_bits`` nats per latent dim are "free"
  (not penalised), which prevents posterior collapse (the failure mode where the
  VAE ignores ``z`` and every sample looks identical).

The VAE loop also uses three standard-but-effective upgrades:

* **EMA** -- an exponential moving average of the weights is validated and
  exported instead of the raw weights; for generative models this reliably
  smooths out the last-epoch jitter in sample quality.
* **AMP** -- mixed-precision autocast on CUDA, roughly halving step time and
  memory so the same GPU affords a wider model or more epochs.
* **warmup + cosine LR** -- linear warmup then cosine decay to ~0.

**The transfer-learning story** lives at the bottom of this file:
:func:`save_base_checkpoint` freezes a trained base into a versioned directory
(weights + a manifest holding the exact architecture and mappings), and
:func:`fine_tune` rebuilds the model from that manifest, assigns a free style
slot, and trains only the style parameters on a user's own chunks. The result
still exports through the normal ONNX path -- one model, several terrain
styles.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from gt_terrain.blocks import BlockGrouping
from gt_terrain.config import Config
from gt_terrain.data import (
    IGNORE_INDEX,
    BiomeEncoder,
    ChunkGridDataset,
    build_dataset,
    compute_class_weights,
    split_indices,
)
from gt_terrain.models import BaselineVoxelNet, ConditionalTerrainVAE, _FiLM

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


class _EMA:
    """Exponential moving average of a model's weights.

    The decay warms up as ``min(decay, (1 + step) / (10 + step))`` so short runs
    (smoke tests, fine-tunes) still produce a meaningful average instead of a
    barely-moved copy of the initial weights.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.step = 0
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for ema_t, live_t in zip(
            self.model.state_dict().values(), model.state_dict().values()
        ):
            if ema_t.dtype.is_floating_point:
                ema_t.mul_(d).add_(live_t.detach(), alpha=1 - d)
            else:
                ema_t.copy_(live_t)


def _warmup_cosine(epochs: int, warmup: int):
    """LR multiplier schedule: linear warmup, then cosine decay to ~0."""
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        t = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0)))
    return lr_lambda


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


def _fit_vae(
    model: ConditionalTerrainVAE,
    config: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    weight: torch.Tensor,
    air_id: int,
    *,
    lr: float,
    epochs: int,
    patience_limit: int,
    warmup_epochs: int,
    ckpt_name: str,
    beta_for_epoch,
    style_id: int = 0,
    biome_dropout: float = 0.0,
    unknown_biome_id: int | None = None,
    tag: str = "vae",
) -> History:
    """Shared VAE fit loop (used by base training and fine-tuning).

    Trains whatever parameters currently have ``requires_grad=True``, maintains
    an EMA copy, validates the EMA weights at the *target* beta (a stable
    early-stopping signal -- validating at the annealed beta made early epochs
    look artificially good), and saves the best EMA weights to ``ckpt_name``.
    On return the live model holds the best EMA weights.
    """
    device = torch.device(config.device)
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # No weight decay on embeddings, norms, or biases (standard practice). This
    # also matters for correctness during fine-tuning: AdamW's decoupled decay
    # shrinks every optimised tensor each step even without gradient, which
    # would silently erode the frozen BASE style row inside style_embed.
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or "embed" in name or "norm" in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    trainable = decay_params + no_decay_params
    opt = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _warmup_cosine(epochs, warmup_epochs))
    ema = _EMA(model, config.ema_decay)
    hw = config.heightmap_weight

    hist = History([], [], float("inf"), -1)
    patience = 0
    monitor_train = len(val_loader) == 0     # tiny fine-tune sets may have no val split

    for epoch in range(epochs):
        beta = beta_for_epoch(epoch)
        model.train()
        tl = 0.0
        for grid, biome in tqdm(train_loader, desc=f"{tag} e{epoch+1} train", leave=False):
            grid, biome = grid.to(device), biome.to(device)
            if biome_dropout > 0 and unknown_biome_id is not None:
                drop = torch.rand(biome.shape, device=device) < biome_dropout
                biome = torch.where(drop, torch.full_like(biome, unknown_biome_id), biome)
            style = torch.full_like(biome, style_id)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                logits, mu, logvar, height = model(grid, biome, style)
                recon = _voxel_ce(logits, grid, weight)
                kl = _kl_with_free_bits(mu, logvar, config.free_bits)
                hloss = F.l1_loss(height, _target_heightmap(grid, air_id))
                loss = recon + beta * kl + hw * hloss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tl += loss.item()
        sched.step()
        hist.train.append(tl / max(len(train_loader), 1))

        # Validate the EMA weights at the target beta so epochs are comparable.
        ema.model.eval()
        vl = 0.0
        with torch.no_grad():
            for grid, biome in val_loader:
                grid, biome = grid.to(device), biome.to(device)
                style = torch.full_like(biome, style_id)
                logits, mu, logvar, height = ema.model(grid, biome, style)
                recon = _voxel_ce(logits, grid, weight)
                kl = _kl_with_free_bits(mu, logvar, config.free_bits)
                hloss = F.l1_loss(height, _target_heightmap(grid, air_id))
                vl += (recon + config.beta * kl + hw * hloss).item()
        hist.val.append(hist.train[-1] if monitor_train else vl / max(len(val_loader), 1))
        print(f"[{tag}] epoch {epoch+1} beta {beta:.3f} lr {sched.get_last_lr()[0]:.2e} "
              f"train {hist.train[-1]:.4f} val {hist.val[-1]:.4f}")

        if hist.val[-1] < hist.best_val:
            hist.best_val, hist.best_epoch, patience = hist.val[-1], epoch, 0
            _save_best(ema.model, ckpt_name, config)
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"[{tag}] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(
        torch.load(config.artifact_dir / ckpt_name, map_location=device)
    )
    return hist


def train_vae(
    config: Config,
    grids: np.ndarray,
    biome_ids: np.ndarray,
    class_weights: np.ndarray,
    num_classes: int,
    num_biomes: int,
    unknown_biome_id: int | None = None,
) -> tuple[ConditionalTerrainVAE, History]:
    """Train the conditional VAE from scratch; saves ``vae.pth``.

    Pass ``unknown_biome_id`` (``BiomeEncoder.unknown_id``) to enable biome
    dropout, which trains the UNKNOWN biome row on generic terrain -- that is
    what lets fine-tuning data with unrecognised biomes still work.
    """
    set_seed(config.seed)
    device = torch.device(config.device)
    train_loader, val_loader, _ = make_loaders(grids, biome_ids, config)

    model = ConditionalTerrainVAE(num_classes, num_biomes, config).to(device)
    weight = torch.tensor(class_weights, device=device)
    air_id = BlockGrouping().group_to_idx["AIR"]

    def beta_for_epoch(epoch: int) -> float:
        return config.beta * min(1.0, (epoch + 1) / max(config.kl_anneal_epochs, 1))

    hist = _fit_vae(
        model, config, train_loader, val_loader, weight, air_id,
        lr=config.lr,
        epochs=config.epochs,
        patience_limit=config.patience,
        warmup_epochs=config.warmup_epochs,
        ckpt_name="vae.pth",
        beta_for_epoch=beta_for_epoch,
        style_id=0,
        biome_dropout=config.biome_dropout,
        unknown_biome_id=unknown_biome_id,
        tag="vae",
    )
    return model, hist


# --- versioned base checkpoint + fine-tuning ---------------------------------

@dataclass
class FineTuneResult:
    """Everything the export step needs after a fine-tune."""
    model: ConditionalTerrainVAE
    history: History
    style_name: str
    style_id: int
    styles: dict[str, int]            # full registry, e.g. {"BASE": 0, "myworld": 1}
    config: Config
    grouping: BlockGrouping
    biome_encoder: BiomeEncoder
    base_dir: Path = field(default=None)


def save_base_checkpoint(
    model: ConditionalTerrainVAE,
    config: Config,
    grouping: BlockGrouping,
    biome_encoder: BiomeEncoder,
    dataset_chunks: int | None = None,
) -> Path:
    """Freeze a trained base model into ``artifact_dir/base_<version>/``.

    The directory is self-contained: weights plus a manifest recording the
    exact architecture and the mappings the weights were trained against.
    :func:`fine_tune` consumes it and never trusts the caller's config for
    architecture -- mismatched fine-tunes are impossible by construction.
    """
    base_dir = Path(config.artifact_dir) / f"base_{config.base_version}"
    base_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), base_dir / "vae_base.pth")

    manifest = {
        "format": 1,
        "version": config.base_version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "num_classes": model.num_classes,
        "num_biomes": biome_encoder.num_biomes,
        "arch": {
            "base_channels": config.base_channels,
            "biome_embed_dim": config.biome_embed_dim,
            "latent_channels": config.latent_channels,
            "latent_grid": list(config.latent_grid),
            "style_dim": config.style_dim,
            "max_custom_styles": config.max_custom_styles,
        },
        "geometry": {
            "chunk_width": config.chunk_width,
            "chunk_height": config.chunk_height,
            "chunk_depth": config.chunk_depth,
            "min_y": config.min_y,
        },
        "biome_mapping": biome_encoder.name_to_id,
        "block_groups": grouping.groups,
        "dataset_chunks": dataset_chunks,
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    styles_path = base_dir / "style_mapping.json"
    if not styles_path.exists():
        styles_path.write_text(json.dumps({"BASE": 0}, indent=2), encoding="utf-8")
    print(f"[base] checkpoint written to {base_dir}")
    return base_dir


def fine_tune(
    base_dir: str | Path,
    data_dir: str | Path,
    style_name: str,
    config: Config | None = None,
) -> FineTuneResult:
    """Adapt a frozen base model to a user's own terrain as a new style.

    This is the product feature: point it at a base checkpoint directory (from
    :func:`save_base_checkpoint`) and a folder of chunk CSVs exported from any
    server, give the style a name, and it returns a model whose new style slot
    generates that terrain -- the base style is untouched, so one exported
    model serves both.

    * Architecture and mappings come from the base manifest, never from the
      caller's config (mismatches are impossible). The passed ``config`` only
      supplies run settings: ``ft_lr``, ``ft_epochs``, ``ft_unfreeze``,
      ``device``, ``artifact_dir``, ...
    * Biomes the base never saw map to the UNKNOWN row, which biome dropout
      trained on generic terrain.
    * ``config.ft_unfreeze`` picks the adaptation size: ``"style"`` trains only
      the new style row; ``"film"`` (default) also trains the FiLM projections;
      ``"decoder"`` unfreezes the whole decoder + heightmap head for large
      custom datasets. The encoder always stays frozen so the latent space
      keeps matching the ``z ~ N(0, I)`` prior the plugin samples from.
    """
    base_dir = Path(base_dir)
    manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    arch = manifest["arch"]

    cfg = (config or Config()).with_(
        data_dir=Path(data_dir),
        base_channels=arch["base_channels"],
        biome_embed_dim=arch["biome_embed_dim"],
        latent_channels=arch["latent_channels"],
        latent_grid=tuple(arch["latent_grid"]),
        style_dim=arch["style_dim"],
        max_custom_styles=arch["max_custom_styles"],
    )
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    grouping = BlockGrouping(groups=manifest["block_groups"])
    biome_encoder = BiomeEncoder.from_mapping(manifest["biome_mapping"])

    # --- user data, gridded with the BASE mappings ---------------------------
    grids, biome_ids = build_dataset(cfg, grouping, biome_encoder)
    class_weights = compute_class_weights(grids, grouping.num_classes)
    train_loader, val_loader, _ = make_loaders(grids, biome_ids, cfg)
    print(f"[ft] {len(grids)} user chunks loaded for style '{style_name}'")

    # --- rebuild the exact base model and load its weights -------------------
    model = ConditionalTerrainVAE(manifest["num_classes"], manifest["num_biomes"], cfg)
    model.load_state_dict(
        torch.load(base_dir / "vae_base.pth", map_location="cpu")
    )
    model = model.to(device)

    # --- style slot allocation ------------------------------------------------
    styles_path = base_dir / "style_mapping.json"
    styles: dict[str, int] = (
        json.loads(styles_path.read_text(encoding="utf-8"))
        if styles_path.exists() else {"BASE": 0}
    )
    if style_name in styles:
        style_id = styles[style_name]        # re-training an existing style
    else:
        style_id = max(styles.values()) + 1
        if style_id > cfg.max_custom_styles:
            raise ValueError(
                f"All {cfg.max_custom_styles} custom style slots are taken: {styles}. "
                f"Re-use a name to retrain it, or train a new base with more slots."
            )
        styles[style_name] = style_id
    # Start the new style where the base style ended up -- fine-tuning then
    # moves it, rather than starting from an inert zero vector.
    with torch.no_grad():
        model.style_embed.weight[style_id] = model.style_embed.weight[0]

    # --- freeze policy ---------------------------------------------------------
    for p in model.parameters():
        p.requires_grad_(False)
    model.style_embed.weight.requires_grad_(True)   # embedding grads are sparse:
    # only the rows actually used in a batch receive gradient, and fine-tuning
    # only ever uses the new style id -- so the BASE row stays intact.
    if cfg.ft_unfreeze in ("film", "decoder"):
        for m in model.modules():
            if isinstance(m, _FiLM):
                for p in m.parameters():
                    p.requires_grad_(True)
    if cfg.ft_unfreeze == "decoder":
        for p in model.decoder.parameters():
            p.requires_grad_(True)
        for p in model.height_head.parameters():
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ft] unfreeze='{cfg.ft_unfreeze}': {n_train:,} trainable parameters")

    model.style_noise = 0.0                  # learn the style exactly, not a noise cloud

    weight = torch.tensor(class_weights, device=device)
    air_id = grouping.group_to_idx["AIR"]
    hist = _fit_vae(
        model, cfg, train_loader, val_loader, weight, air_id,
        lr=cfg.ft_lr,
        epochs=cfg.ft_epochs,
        patience_limit=cfg.ft_patience,
        warmup_epochs=min(2, cfg.ft_epochs),
        ckpt_name=f"vae_ft_{style_name}.pth",
        beta_for_epoch=lambda _e: cfg.beta,  # no annealing: the model is pre-trained
        style_id=style_id,
        tag=f"ft:{style_name}",
    )

    # Persist the updated registry both in the base dir (slot bookkeeping) and
    # the artifact dir (ships to the plugin next to the ONNX).
    styles_json = json.dumps(styles, indent=2)
    styles_path.write_text(styles_json, encoding="utf-8")
    cfg.ensure_dirs()
    (Path(cfg.artifact_dir) / "style_mapping.json").write_text(styles_json, encoding="utf-8")

    return FineTuneResult(
        model=model, history=hist, style_name=style_name, style_id=style_id,
        styles=styles, config=cfg, grouping=grouping, biome_encoder=biome_encoder,
        base_dir=base_dir,
    )
