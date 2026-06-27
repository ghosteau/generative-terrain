"""
gt_terrain
==========

A clean, leakage-free pipeline for training Minecraft terrain generators.

The package is intentionally small and composable so that the accompanying
notebook (``ml/notebooks/GenerativeTerrain.ipynb``) stays thin -- every heavy
operation lives here, is unit-tested, and is importable from both the notebook
and plain scripts.

Module map
----------
``config``    Runtime configuration (paths, hyper-parameters, device selection).
``blocks``    The single source of truth for block grouping (block -> group id).
``data``      CSV -> voxel-grid conversion, biome encoding, and the dataset.
``models``    The two models: ``BaselineVoxelNet`` and ``ConditionalTerrainVAE``.
``train``     Training loops for both models.
``evaluate``  Accuracy / terrain-plausibility metrics and plots.
``generate``  Sampling new chunks from a trained VAE.
``export``    ONNX export of the decoder plus the JSON mappings the plugin reads.

Design rule (why this package exists)
-------------------------------------
The previous models conditioned each voxel on the block ids of its neighbours
plus ``Is_Surface`` and ``Light_Level`` -- all of which are *properties of the
terrain being generated*. That is data leakage: it is unavailable at generation
time. Everything here conditions only on information that exists *before* terrain
does: position (especially height ``y``), the chunk biome, and -- for the
generative model -- a latent noise vector ``z``.
"""

from gt_terrain.config import Config, default_config

__all__ = ["Config", "default_config"]
