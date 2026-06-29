# GenerativeTerrain — ML pipeline

See the [root README](../README.md) for the full project overview, architecture description, and deployment guide.

This document covers ML-specific details: the package layout, setup, and how to run.

## Package layout

```
gt_terrain/
  config.py       Paths + hyperparameters (env-overridable). Config.tiny() for CPU dry-runs.
  blocks.py       27-group block taxonomy — single source of truth shared with Java.
  data.py         CSV → voxel grids (leakage-free, vectorised). Dataset + augmentation.
  models.py       BaselineVoxelNet (deterministic floor) + ConditionalTerrainVAE.
                  VAE has a spatial latent grid + 3D U-Net decoder + internal heightmap
                  stage. Single ONNX forward pass: (z, biome_id) → logits.
  train.py        Training loops. VAE uses KL annealing, free bits, and heightmap L1 loss.
  evaluate.py     Per-group accuracy, vertical block-by-height profiles, sample diversity.
  generate.py     Sample chunks from the trained VAE (with optional post-processing).
  postprocess.py  Morphological cleanup (despeckle / fill pinholes) + procedural ore scatter.
  export.py       ONNX decoder export + JSON mapping files the plugin reads.
notebooks/
  GenerativeTerrain.ipynb   The driver notebook. Run top to bottom.
  legacy/                   Six superseded notebooks, kept for reference.
tests/
  test_data.py    pytest suite; includes the no-leakage invariant.
```

## Setup

```bash
# From the ml/ directory:
pip install -r requirements.txt
```

**GPU training (RTX 4070 or similar):** install the CUDA torch build first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Configure data paths

```powershell
# Windows PowerShell
$env:GT_DATA_DIR    = "C:\path\to\Server\block_dataset"
$env:GT_ARTIFACT_DIR = "C:\path\to\outputs"
```

Defaults: `GT_DATA_DIR` points at the PaperMC server export folder; `GT_ARTIFACT_DIR` is `ml/artifacts/`.

## Run

- **Notebook:** open `notebooks/GenerativeTerrain.ipynb` and run top to bottom.
- **CPU smoke test:** swap `cfg = Config()` for `cfg = Config.tiny()` in the notebook.
- **Tests:** `python -m pytest tests/`

## Output → plugin

The export step writes three files to `Config.artifact_dir`:

```
terrain_vae_decoder.onnx
block_group_mapping.json
biome_id_mapping.json
```

Copy all three into `<server>/plugins/GenerativeTerrain/`, then run `/modelgenerateterrain` in-game.

## Data limitations

- The VAE needs a lot of data. Speckle and biome homogeneity are symptoms of too few chunks.
- Use `/grabchunkarea <radius>` for bulk collection and `/grabbiome <biome>` for rare biomes.
- Biomes with fewer than ~20 chunks will be undertrained regardless of sampler weights.
