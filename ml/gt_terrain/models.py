"""
Models.

Two models, sharing one decoder body so they are directly comparable:

* :class:`BaselineVoxelNet` -- deterministic. Conditions on (position, biome)
  only. It can only learn the *average* terrain for a biome, so it produces
  smooth, layered terrain (correct block-by-height, no hills/caves/variety).
  Its job is to prove the pipeline is sound and set an accuracy floor.

* :class:`ConditionalTerrainVAE` -- generative. An encoder compresses a real
  chunk into a latent ``z``; the decoder reconstructs it from ``z`` + biome +
  position. At generation time we sample ``z ~ N(0, I)``, so different ``z``
  give different terrain for the same biome -- that is where variety comes from.

Why a "broadcast latent" decoder (no transposed convs)? The chunk is
16 x 384 x 16 and 384 is awkward to up-sample to exactly. Instead the decoder
builds a full-resolution coordinate volume, broadcasts ``z`` and the biome
embedding across every voxel, concatenates them, and runs a 3D conv stack. This
works for any chunk geometry and exports cleanly to ONNX.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gt_terrain.config import Config


def _num_groups(channels: int) -> int:
    """Largest GroupNorm group count (<=8) that divides ``channels``."""
    return next(g for g in (8, 4, 2, 1) if channels % g == 0)


def coordinate_features(
    batch: int, x: int, y: int, z: int, device: torch.device | str
) -> torch.Tensor:
    """Normalised (x, y, z) position channels, shape ``[B, 3, X, Y, Z]``.

    Height (the Y channel) is the single most informative input for terrain, so
    making it an explicit, normalised signal matters more than any architecture
    choice here.
    """
    xs = torch.linspace(0, 1, x, device=device)
    ys = torch.linspace(0, 1, y, device=device)
    zs = torch.linspace(0, 1, z, device=device)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pos = torch.stack([gx, gy, gz], dim=0)          # [3, X, Y, Z]
    return pos.unsqueeze(0).expand(batch, -1, -1, -1, -1)


class _ResBlock3D(nn.Module):
    """Pre-norm 3x3x3 residual conv block."""

    def __init__(self, channels: int):
        super().__init__()
        groups = _num_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        r = h
        h = self.conv1(F.gelu(self.norm1(h)))
        h = self.conv2(F.gelu(self.norm2(h)))
        return h + r


class _DecoderBody(nn.Module):
    """Per-voxel conditional decoder: conditioning volume -> class logits.

    Shared by both models. Input is whatever conditioning channels the caller
    assembled (position, biome, and optionally a broadcast latent); output is
    ``[B, num_classes, X, Y, Z]``.
    """

    def __init__(self, in_channels: int, hidden: int, num_classes: int, num_blocks: int = 4):
        super().__init__()
        self.stem = nn.Conv3d(in_channels, hidden, 3, padding=1)
        self.blocks = nn.ModuleList(_ResBlock3D(hidden) for _ in range(num_blocks))
        self.head = nn.Sequential(
            nn.GroupNorm(_num_groups(hidden), hidden),
            nn.GELU(),
            nn.Conv3d(hidden, num_classes, 1),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        h = self.stem(cond)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


class _ConvBlock(nn.Module):
    """Two 3x3x3 convs with GroupNorm + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g = _num_groups(out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(g, out_ch), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(g, out_ch), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class _UNetDecoder(nn.Module):
    """3D U-Net mapping a conditioning volume to class logits.

    The two down/up levels give the model a chunk-scale receptive field, so it
    can form coherent large structures (hills, cave systems) rather than
    predicting each voxel from a tiny local window. Down/up sampling uses
    pooling + trilinear interpolation (with recorded sizes) so it works for any
    chunk geometry and exports cleanly to ONNX.
    """

    def __init__(self, in_channels: int, base: int, num_classes: int):
        super().__init__()
        self.in_conv = _ConvBlock(in_channels, base)
        self.enc1 = _ConvBlock(base, base)
        self.enc2 = _ConvBlock(base, base * 2)
        self.bottleneck = _ConvBlock(base * 2, base * 2)
        self.dec2 = _ConvBlock(base * 2 + base * 2, base * 2)
        self.dec1 = _ConvBlock(base * 2 + base, base)
        self.head = nn.Conv3d(base, num_classes, 1)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(cond)
        e1 = self.enc1(x0)                                   # full res (skip)
        e2 = self.enc2(F.max_pool3d(e1, 2))                 # /2 (skip)
        b = self.bottleneck(F.max_pool3d(e2, 2))            # /4
        d2 = F.interpolate(b, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))          # /2
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))          # full res
        return self.head(d1)


class BaselineVoxelNet(nn.Module):
    """Deterministic (position, biome) -> block-group logits."""

    def __init__(self, num_classes: int, num_biomes: int, config: Config):
        super().__init__()
        self.config = config
        self.biome_embed = nn.Embedding(num_biomes, config.biome_embed_dim)
        in_ch = 3 + config.biome_embed_dim
        self.decoder = _DecoderBody(in_ch, config.base_channels, num_classes)

    def forward(self, biome_id: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        x, y, z = shape
        b = biome_id.shape[0]
        device = biome_id.device
        pos = coordinate_features(b, x, y, z, device)                 # [B,3,X,Y,Z]
        be = self.biome_embed(biome_id)                              # [B, E]
        be = be[:, :, None, None, None].expand(-1, -1, x, y, z)      # [B,E,X,Y,Z]
        cond = torch.cat([pos, be], dim=1)
        return self.decoder(cond)


class _Encoder(nn.Module):
    """Real chunk (+ biome) -> SPATIAL latent (mu, logvar) of shape [B, Cz, lx, ly, lz].

    Instead of pooling the whole chunk to a single vector, we pool to a small 3D
    grid. Each latent cell summarises a region of the chunk, so the decoder can
    reconstruct (and later sample) terrain that differs from place to place.
    """

    def __init__(self, num_classes: int, num_biomes: int, config: Config):
        super().__init__()
        c = config.base_channels
        self.latent_grid = config.latent_grid
        self.block_embed = nn.Embedding(num_classes, c)
        self.biome_embed = nn.Embedding(num_biomes, config.biome_embed_dim)
        self.down = nn.Sequential(
            nn.Conv3d(c + config.biome_embed_dim, c, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(c, c * 2, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv3d(c * 2, c * 4, 4, stride=2, padding=1), nn.GELU(),
        )
        # Pool to the coarse latent grid, then 1x1 conv to 2*Cz (mu + logvar).
        self.pool = nn.AdaptiveAvgPool3d(config.latent_grid)
        self.to_latent = nn.Conv3d(c * 4, config.latent_channels * 2, 1)

    def forward(self, grid: torch.Tensor, biome_id: torch.Tensor):
        # grid: [B,X,Y,Z] longs with -1 for "ignore"; clamp so embedding is valid.
        b, x, y, z = grid.shape
        emb = self.block_embed(grid.clamp(min=0))               # [B,X,Y,Z,C]
        emb = emb.permute(0, 4, 1, 2, 3)                        # [B,C,X,Y,Z]
        be = self.biome_embed(biome_id)[:, :, None, None, None].expand(-1, -1, x, y, z)
        h = torch.cat([emb, be], dim=1)
        h = self.down(h)
        h = self.pool(h)                                        # [B, c*4, lx, ly, lz]
        mu, logvar = self.to_latent(h).chunk(2, dim=1)          # each [B, Cz, lx, ly, lz]
        return mu, logvar


class ConditionalTerrainVAE(nn.Module):
    """Conditional VAE with a spatial latent; generation = sampling ``z``.

    The latent ``z`` is a coarse 3D grid (``config.vae_latent_shape``) that the
    decoder upsamples to full resolution. Combined with the U-Net decoder, this
    is what lets generated terrain have hills and caves rather than flat layers.
    """

    def __init__(self, num_classes: int, num_biomes: int, config: Config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = _Encoder(num_classes, num_biomes, config)
        self.biome_embed = nn.Embedding(num_biomes, config.biome_embed_dim)
        in_ch = 3 + config.latent_channels + config.biome_embed_dim
        self.decoder = _UNetDecoder(in_ch, config.base_channels, num_classes)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(
        self, z: torch.Tensor, biome_id: torch.Tensor, shape: tuple[int, int, int]
    ) -> torch.Tensor:
        # z: [B, Cz, lx, ly, lz] -> upsample to full resolution and condition on it.
        x, y, z_dim = shape
        b = z.shape[0]
        pos = coordinate_features(b, x, y, z_dim, z.device)
        zt = F.interpolate(z, size=(x, y, z_dim), mode="trilinear", align_corners=False)
        be = self.biome_embed(biome_id)[:, :, None, None, None].expand(-1, -1, x, y, z_dim)
        cond = torch.cat([pos, zt, be], dim=1)
        return self.decoder(cond)

    def forward(self, grid: torch.Tensor, biome_id: torch.Tensor):
        mu, logvar = self.encoder(grid, biome_id)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, biome_id, grid.shape[1:])
        return logits, mu, logvar


class VAEDecoderForExport(nn.Module):
    """Thin ONNX-export wrapper: ``(z, biome_id) -> logits`` at fixed geometry.

    The Java plugin samples ``z``, picks the biome, and runs exactly this graph.
    """

    def __init__(self, vae: ConditionalTerrainVAE, shape: tuple[int, int, int]):
        super().__init__()
        self.vae = vae
        self.shape = shape

    def forward(self, z: torch.Tensor, biome_id: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z, biome_id, self.shape)
