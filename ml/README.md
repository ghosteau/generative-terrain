# GenerativeTerrain — ML pipeline

See the [root README](../README.md) for the full project overview, architecture description, and deployment guide.

This document covers ML-specific details: the package layout, setup, and how to run.

## Package layout

```
gt_terrain/
  config.py       Paths + hyperparameters (env-overridable). Config.tiny() for CPU dry-runs.
  blocks.py       27-group block taxonomy — single source of truth shared with Java.
  data.py         CSV → voxel grids (leakage-free, vectorised). Dataset + augmentation.
                  BiomeEncoder reserves an UNKNOWN row for foreign-world biomes.
  models.py       BaselineVoxelNet (deterministic floor) + ConditionalTerrainVAE.
                  VAE: spatial latent grid + 3-level 3D U-Net + internal heightmap stage
                  + FiLM style conditioning. One ONNX pass: (z, biome_id, style_id) → logits.
  train.py        Training loops (KL annealing, free bits, EMA, AMP, warmup+cosine LR,
                  biome dropout) + save_base_checkpoint() + fine_tune().
  evaluate.py     Per-group accuracy, vertical block-by-height profiles, sample diversity.
  generate.py     Sample chunks from the trained VAE (per style, optional post-processing).
  postprocess.py  Morphological cleanup (despeckle / fill pinholes) + procedural ore scatter.
  export.py       ONNX decoder export + JSON mapping files the plugin reads.
notebooks/
  GenerativeTerrain.ipynb   Base-training driver. Run top to bottom; ends by freezing
                            a versioned base checkpoint (artifacts/base_v1/).
  FineTune.ipynb            Adapt the frozen base to your own terrain as a new style.
  legacy/                   Six superseded notebooks, kept for reference.
tests/
  test_data.py      pytest; includes the no-leakage invariant.
  test_finetune.py  pytest; style system + fine-tune freeze guarantees + ONNX inputs.
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

The export step writes four files to `Config.artifact_dir`:

```
terrain_vae_decoder.onnx
block_group_mapping.json
biome_id_mapping.json
style_mapping.json
```

Copy all four into `<server>/plugins/GenerativeTerrain/`, then run
`/generateterrain` in-game (`/generateterrain style <name>` for a fine-tuned
style).

## Fine-tuning

`GenerativeTerrain.ipynb` ends by freezing the trained model into
`artifacts/base_v1/` (weights + manifest). `FineTune.ipynb` consumes that
directory plus a folder of your own chunk CSVs and trains a new style
(~1.8% of the parameters, minutes on GPU) without touching the base weights —
`tests/test_finetune.py` asserts they come out bit-identical.

## Data limitations

- The VAE needs a lot of data. Speckle and biome homogeneity are symptoms of too few chunks.
- Use `/grabchunkarea <radius>` for bulk collection and `/grabbiome <biome>` for rare biomes.
- Biomes with fewer than ~20 chunks will be undertrained regardless of sampler weights.
