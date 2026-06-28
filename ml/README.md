# GenerativeTerrain — ML pipeline

Training code for the Minecraft terrain generator. Everything lives in the
`gt_terrain` package; the notebook (`notebooks/GenerativeTerrain.ipynb`) is a
thin driver that narrates the stages.

## The core idea (read this first)

The previous models conditioned each voxel on the **block IDs of its neighbours**
plus `Is_Surface` and `Light_Level`. Those describe the terrain *being
generated*, so the model only ever learned "fill a hole given its neighbours" and
could not generate from scratch — at generation time those inputs don't exist.

This pipeline conditions only on what's available **before** terrain exists:

| Input            | Why it's allowed                              |
|------------------|-----------------------------------------------|
| position (x,y,z) | known a priori; height `y` is the key signal  |
| chunk biome      | chosen before generating                      |
| latent noise `z` | the source of variety (generative model only) |

## Layout

```
gt_terrain/
  config.py     Paths + hyper-parameters (env-overridable). Config.tiny() for CPU dry-runs.
  blocks.py     27-group block taxonomy — the single source of truth (Python + Java share it).
  data.py       CSV -> voxel grids (leakage-free, vectorised), dataset + augmentation.
  models.py     BaselineVoxelNet (deterministic floor) and ConditionalTerrainVAE
                (generative; SPATIAL latent + U-Net decoder so terrain can have
                hills and caves rather than flat layers).
  train.py      Training loops (class-weighted CE; VAE adds KL annealing + free bits).
  evaluate.py   Accuracy + terrain-plausibility (vertical profiles), diversity, plots.
  generate.py   Sample chunks from the VAE; write a block-name CSV.
  export.py     ONNX decoder export + the JSON mappings the plugin reads.
notebooks/
  GenerativeTerrain.ipynb   The driver notebook.
  legacy/                   The six superseded notebooks, kept for reference.
tests/          pytest — includes the no-leakage invariant.
```

## Setup

```bash
# From the ml/ directory:
pip install -r requirements.txt
```

**GPU training:** `requirements.txt` pins the CPU build of torch so it installs
anywhere. To train on your RTX 4070, install the CUDA build *instead* first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Configure the data location

Raw chunk CSVs (from `/grabchunkdata`) are read from `Config.data_dir`, which
defaults to the PaperMC server export folder. Override without editing code:

```bash
# Windows PowerShell
$env:GT_DATA_DIR = "C:\path\to\block_dataset"
$env:GT_ARTIFACT_DIR = "C:\path\to\outputs"
```

## Run

* **Notebook:** open `notebooks/GenerativeTerrain.ipynb` and run top to bottom.
* **Tests:** `python -m pytest tests/`
* **Quick CPU sanity check:** in the notebook swap `cfg = Config()` for
  `cfg = Config.tiny()`.

## Output → plugin

The export step writes three files to `Config.artifact_dir`:

* `terrain_vae_decoder.onnx`
* `block_group_mapping.json`
* `biome_id_mapping.json`

Copy all three into `<server>/plugins/GenerativeTerrain/`, then run
`/generateterrain` in-game.

## Current limitations / next steps

* **Data:** ~230 chunks today. This is the biggest constraint; the VAE is
  data-hungry. Collect more quickly in-game with **`/grabchunkarea <radius>`**,
  which exports a whole square of chunks at once (one CSV each).
* **Per-chunk independence:** chunks are generated in isolation, so edges between
  adjacent generated chunks won't line up yet.
* **Scale with data:** raise `base_channels`, `latent_channels`, the
  `latent_grid` resolution, and `epochs` as the dataset grows. The VAE needs
  enough epochs to converge -- undertrained it produces speckle (stray blocks);
  watch the vertical-profile plot to judge convergence.
