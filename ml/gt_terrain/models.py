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
  The decoder is additionally conditioned on a **style** embedding via FiLM;
  style 0 is the base style and the reserved extra slots are what
  ``train.fine_tune`` trains on a user's own chunks (transfer learning).

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


class _FiLM(nn.Module):
    """Feature-wise linear modulation of a feature map by a style vector.

    ``h * (1 + scale) + shift`` with scale/shift projected from the style
    vector; zero-initialised so it is an exact identity at the start of
    training. This is the transfer-learning mechanism: fine-tuning trains a new
    style vector (plus, optionally, these projections), which re-modulates
    every level of the otherwise-frozen decoder -- a global stylistic change
    from an adapter-sized set of parameters.
    """

    def __init__(self, style_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(style_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(style).chunk(2, dim=1)
        while scale.dim() < h.dim():                    # [B,C] -> [B,C,1,(1),1]
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return h * (1 + scale) + shift


class _ConvBlock(nn.Module):
    """Residual double 3x3x3 conv (GroupNorm + GELU) with FiLM style modulation."""

    def __init__(self, in_ch: int, out_ch: int, style_dim: int):
        super().__init__()
        g = _num_groups(out_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(g, out_ch)
        self.film = _FiLM(style_dim, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.norm1(self.conv1(x)))
        h = self.film(h, style)
        h = F.gelu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class _UNetDecoder(nn.Module):
    """3D U-Net mapping a conditioning volume (+ style vector) to class logits.

    Three down/up levels give the model a chunk-scale receptive field: at the
    bottleneck the 16 x 384 x 16 chunk is a 2 x 48 x 2 volume, so horizontal
    context is effectively global and vertical context spans whole mountains.
    Down/up sampling uses pooling + trilinear interpolation (with recorded
    sizes) so it works for any even chunk geometry and exports cleanly to ONNX.
    Every block is FiLM-modulated by the style vector.
    """

    def __init__(self, in_channels: int, base: int, num_classes: int, style_dim: int):
        super().__init__()
        self.in_conv = _ConvBlock(in_channels, base, style_dim)
        self.enc1 = _ConvBlock(base, base, style_dim)
        self.enc2 = _ConvBlock(base, base * 2, style_dim)
        self.enc3 = _ConvBlock(base * 2, base * 4, style_dim)
        self.bottleneck = _ConvBlock(base * 4, base * 4, style_dim)
        self.dec3 = _ConvBlock(base * 4 + base * 4, base * 4, style_dim)
        self.dec2 = _ConvBlock(base * 4 + base * 2, base * 2, style_dim)
        self.dec1 = _ConvBlock(base * 2 + base, base, style_dim)
        self.head = nn.Conv3d(base, num_classes, 1)

    def forward(self, cond: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(cond, style)
        e1 = self.enc1(x0, style)                                   # full res (skip)
        e2 = self.enc2(F.max_pool3d(e1, 2), style)                  # /2 (skip)
        e3 = self.enc3(F.max_pool3d(e2, 2), style)                  # /4 (skip)
        b = self.bottleneck(F.max_pool3d(e3, 2), style)             # /8
        d3 = F.interpolate(b, size=e3.shape[2:], mode="trilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), style)           # /4
        d2 = F.interpolate(d3, size=e2.shape[2:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), style)           # /2
        d1 = F.interpolate(d2, size=e1.shape[2:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), style)           # full res
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
        # Clamp keeps exp(logvar) finite under fp16 autocast and bounds the KL.
        return mu, logvar.clamp(-8.0, 8.0)


class _HeightmapHead(nn.Module):
    """Predict a per-column surface height in [0, 1] from the latent + biome + style.

    This is the first 'stage': before deciding each voxel, the model commits to
    where the surface is for each (x, z) column. The voxel decoder then receives
    a signed-distance-to-surface channel, which removes the up/down ambiguity
    that made the model carve out air in hilly biomes. The style vector
    modulates it via FiLM -- terrain style is largely a statement about the
    surface (mountains vs plains), so the style hook here matters as much as
    the ones in the voxel U-Net.
    """

    def __init__(self, latent_channels: int, biome_embed_dim: int, hidden: int, style_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(latent_channels + biome_embed_dim, hidden, 3, padding=1)
        self.film = _FiLM(style_dim, hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(
        self,
        z: torch.Tensor,
        be2d: torch.Tensor,
        style: torch.Tensor,
        out_xz: tuple[int, int],
    ) -> torch.Tensor:
        zc = z.mean(dim=3)                              # collapse latent Y -> [B, Cz, lx, lz]
        h = F.gelu(self.conv1(torch.cat([zc, be2d], dim=1)))
        h = self.film(h, style)
        h = self.out(F.gelu(self.conv2(h)))             # [B, 1, lx, lz]
        h = F.interpolate(h, size=out_xz, mode="bilinear", align_corners=False)
        return torch.sigmoid(h)                         # [B, 1, X, Z] in [0,1]


class ConditionalTerrainVAE(nn.Module):
    """Conditional VAE with a spatial latent, heightmap stage, and style slots.

    The latent ``z`` is a coarse 3D grid (``config.vae_latent_shape``). The
    decoder (1) predicts a per-column surface heightmap from ``z`` + biome +
    style, then (2) fills the volume conditioned on the upsampled latent,
    position, biome, style (via FiLM), and a signed-distance-to-surface
    channel. Both stages live inside one graph, so generation is a single ONNX
    pass with the ``(z, biome_id, style_id) -> logits`` interface.

    **Styles** are the transfer-learning hook. Style 0 is the vanilla base
    style; ``config.max_custom_styles`` extra rows are reserved and stay
    untouched during base training. ``train.fine_tune`` later assigns a free
    row to a user's terrain and trains it (plus the FiLM projections) on their
    chunks while the rest of the network stays frozen -- one model, several
    terrain styles, selectable at generation time.
    """

    def __init__(self, num_classes: int, num_biomes: int, config: Config):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.encoder = _Encoder(num_classes, num_biomes, config)
        self.biome_embed = nn.Embedding(num_biomes, config.biome_embed_dim)
        # Row 0 = base style; the rest are reserved fine-tuning slots. All rows
        # start at zero so, with the zero-initialised FiLM layers, styles are
        # inert until training gives them meaning.
        self.style_embed = nn.Embedding(1 + config.max_custom_styles, config.style_dim)
        nn.init.zeros_(self.style_embed.weight)
        # Mutable on purpose (train.fine_tune sets it to 0 so the custom style
        # is learned exactly, not through a noise cloud).
        self.style_noise = config.style_noise
        self.height_head = _HeightmapHead(
            config.latent_channels, config.biome_embed_dim, config.base_channels,
            config.style_dim,
        )
        # +1 channel for the signed distance to the predicted surface.
        in_ch = 3 + config.latent_channels + config.biome_embed_dim + 1
        self.decoder = _UNetDecoder(in_ch, config.base_channels, num_classes, config.style_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _style_vector(self, style_id: torch.Tensor | None, batch: int, device) -> torch.Tensor:
        if style_id is None:
            style_id = torch.zeros(batch, dtype=torch.long, device=device)
        s = self.style_embed(style_id)                                   # [B, S]
        if self.training and self.style_noise > 0:
            # Perturbing the style during base training keeps the FiLM response
            # smooth around the base style, so fine-tuned styles land in a
            # well-conditioned region instead of untrained territory.
            s = s + torch.randn_like(s) * self.style_noise
        return s

    def decode(
        self,
        z: torch.Tensor,
        biome_id: torch.Tensor,
        shape: tuple[int, int, int],
        style_id: torch.Tensor | None = None,
        return_height: bool = False,
    ):
        # z: [B, Cz, lx, ly, lz]. Stage 1: heightmap. Stage 2: fill.
        x, y, z_dim = shape
        b = z.shape[0]
        be = self.biome_embed(biome_id)                                  # [B, E]
        s = self._style_vector(style_id, b, z.device)                    # [B, S]

        # Stage 1 -- per-column surface height in [0,1].
        be2d = be[:, :, None, None].expand(-1, -1, z.shape[2], z.shape[4])
        height = self.height_head(z, be2d, s, (x, z_dim))                # [B, 1, X, Z]

        # Signed distance to surface: negative below ground, positive above.
        y_norm = torch.linspace(0, 1, y, device=z.device).view(1, 1, 1, y, 1)
        dist = y_norm - height.unsqueeze(3)                             # [B, 1, X, Y, Z]

        # Stage 2 -- fill the volume.
        pos = coordinate_features(b, x, y, z_dim, z.device)
        zt = F.interpolate(z, size=(x, y, z_dim), mode="trilinear", align_corners=False)
        bev = be[:, :, None, None, None].expand(-1, -1, x, y, z_dim)
        cond = torch.cat([pos, zt, bev, dist], dim=1)
        logits = self.decoder(cond, s)
        return (logits, height) if return_height else logits

    def forward(
        self,
        grid: torch.Tensor,
        biome_id: torch.Tensor,
        style_id: torch.Tensor | None = None,
    ):
        mu, logvar = self.encoder(grid, biome_id)
        z = self.reparameterize(mu, logvar)
        logits, height = self.decode(
            z, biome_id, grid.shape[1:], style_id=style_id, return_height=True
        )
        return logits, mu, logvar, height


class VAEDecoderForExport(nn.Module):
    """Thin ONNX-export wrapper: ``(z, biome_id, style_id) -> logits`` at fixed geometry.

    The Java plugin samples ``z``, picks the biome and style, and runs exactly
    this graph. Exported in eval mode, so no style noise is traced in.
    """

    def __init__(self, vae: ConditionalTerrainVAE, shape: tuple[int, int, int]):
        super().__init__()
        self.vae = vae
        self.shape = shape

    def forward(
        self, z: torch.Tensor, biome_id: torch.Tensor, style_id: torch.Tensor
    ) -> torch.Tensor:
        return self.vae.decode(z, biome_id, self.shape, style_id=style_id)
