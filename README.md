# GenerativeTerrain

A Minecraft PaperMC plugin and PyTorch training pipeline that uses a **conditional VAE** to generate terrain chunk-by-chunk from a sampled latent noise vector, a biome ID, and a **style ID** — where styles are the transfer-learning hook: anyone can **fine-tune the pretrained base on their own world's chunks** and get a generator for their terrain, in minutes, without retraining the base.

The project has a single hard constraint: inference must be a fast, single-pass ONNX call inside the Java plugin. This rules out diffusion and autoregressive models; the architecture is designed around that limit.

---

## How it works

### The key insight: no data leakage

Earlier attempts conditioned each voxel on its 6-neighbour block IDs, `Is_Surface`, and `Light_Level`. These are properties of the terrain being generated — at generation time they don't exist. The model learned "fill a hole given its neighbours," which is useless for generation from scratch.

The current pipeline conditions **only on information available before terrain exists**:

| Input | Why it's allowed |
|---|---|
| Position (x, y, z) | Known a priori. The Y channel is the most informative single signal for terrain. |
| Chunk biome | Chosen before generating. |
| Latent noise `z` | The source of variety. Different `z` → different terrain for the same biome. |
| Style ID | Chosen before generating. 0 = vanilla base style; higher ids = fine-tuned custom styles. |

### Architecture

```
                    Training                              Generation
                       ↓                                     ↓
Real chunk (16×384×16) → Encoder → μ, logσ²     z ~ N(0, I) (spatial grid)
                                      ↓                      ↓
                                 Reparameterize → z ──→ Decoder  ←— biome ID, style ID
                                                          ↓
                                             Stage 1: HeightmapHead (FiLM-styled)
                                             (per-column surface in [0,1])
                                                          ↓
                                             signed-distance-to-surface channel
                                                          ↓
                                             Stage 2: 3D U-Net fills volume
                                             (FiLM style modulation at every block)
                                                          ↓
                                             Logits [27, 16, 384, 16]
```

**Spatial latent** — `z` is a coarse 3D grid `(8, 4, 16, 4)` rather than a single broadcast vector. A broadcast vector can only produce the average column per biome (flat terrain, no hills, no caves). The spatial grid lets each region of the chunk vary independently.

**Heightmap stage** — before filling every voxel the decoder predicts a per-column surface height from `z` + biome + style. That surface feeds a signed-distance channel (negative below ground, positive above) to the U-Net. This removes the up/down ambiguity that caused the model to carve out air in hilly biomes.

**3D U-Net decoder** — three down/up levels with skip connections and residual blocks; at the bottleneck the chunk is a 2×48×2 volume, so horizontal context is effectively global and vertical context spans whole mountains. ~4.6M parameters at the default width.

**Style embedding + FiLM (the transfer-learning mechanism)** — every U-Net block and the heightmap head is modulated by a style vector via FiLM (per-channel scale/shift, zero-initialised). Style 0 is the vanilla base; 8 extra slots are reserved for custom styles. During base training the style vector is jittered with Gaussian noise so the FiLM response is smooth around the base — which is exactly what lets fine-tuning later move through well-conditioned territory. A reserved `UNKNOWN` biome row (trained via biome dropout) handles fine-tuning data from worlds whose biomes the base never saw.

**Post-processing** — the model predicts terrain shape; two deterministic steps are applied afterwards (matching how vanilla Minecraft works):
- *Morphological cleanup*: remove single floating blocks and fill single-voxel pinholes. Keeps real caves; removes speckle.
- *Procedural ore scatter*: place coal/iron/diamond/etc. into stone and deepslate at vanilla-ish Y ranges and probabilities. Learning rare scattered classes from a VAE is unreliable; doing it procedurally is controllable and always works.

Both steps are mirrored exactly in the Java plugin (`TerrainPostProcessor.java`).

**ONNX export** — only the decoder is exported: `(z, biome_id, style_id) → logits`. The encoder is training-only. All style rows are baked into one graph, so a single exported model serves the base style and every fine-tuned style. The Java plugin samples `z ~ N(0, I)`, reads the z shape and the declared inputs directly from the ONNX graph (older two-input models keep working), and runs a single inference call.

---

## Repository layout

```
GenerativeTerrain/
├── src/main/java/com/ghosteau/generativeterrain/
│   ├── GenerativeTerrain.java          Main plugin class + command registration
│   └── commands/
│       ├── setDataPath.java            /setdatapath — set where CSVs are exported
│       ├── grabChunkData.java          /grabchunkdata — export current chunk
│       ├── grabChunkArea.java          /grabchunkarea — export a square area
│       ├── grabBiome.java              /grabbiome — targeted rare-biome collection
│       ├── grabBlock.java              /grabblock — targeted rare-block collection
│       ├── ChunkSearchTask.java        Shared spiral search + async write engine
│       ├── ChunkDataExtractor.java     Block-reading and CSV-building logic
│       ├── modelGenerateTerrain.java   /modelgenerateterrain — ONNX inference + placement
│       └── TerrainPostProcessor.java   Java-side morphological clean + ore scatter
├── ml/
│   ├── gt_terrain/
│   │   ├── config.py       All paths and hyperparameters (env-overridable)
│   │   ├── blocks.py       27-group block taxonomy — single source of truth
│   │   ├── data.py         CSV → voxel grids, leakage-free, dataset + augmentation
│   │   ├── models.py       BaselineVoxelNet + ConditionalTerrainVAE (FiLM styles)
│   │   ├── train.py        Training loops, versioned base checkpoint, fine_tune()
│   │   ├── evaluate.py     Accuracy, vertical profiles, sample diversity
│   │   ├── generate.py     Sample chunks from the trained VAE (per style)
│   │   ├── postprocess.py  Morphological clean + procedural ore scatter
│   │   └── export.py       ONNX export + JSON mapping files
│   ├── notebooks/
│   │   ├── GenerativeTerrain.ipynb   Base-training driver notebook
│   │   ├── FineTune.ipynb            Fine-tune the base on your own terrain
│   │   └── legacy/                   Six superseded notebooks kept for reference
│   ├── tests/
│   │   ├── test_data.py       pytest: no-leakage invariant + pipeline
│   │   └── test_finetune.py   pytest: styles, fine-tune freeze guarantees, ONNX
│   ├── requirements.txt
│   └── README.md           ML-specific setup and run guide
├── plugins_server/         PaperMC dependency jars
└── pom.xml
```

---

## Setup

### Python (training)

```powershell
cd ml
pip install -r requirements.txt
```

**GPU training (RTX 4070 or similar):** install the CUDA torch build first, then the rest of requirements:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Configure data paths** via environment variables (no hard-coded paths):

```powershell
$env:GT_DATA_DIR    = "C:\path\to\Server\block_dataset"   # where CSVs are exported
$env:GT_ARTIFACT_DIR = "C:\path\to\outputs"               # where model + mappings land
```

If these aren't set, `GT_DATA_DIR` defaults to the PaperMC server export folder used by the plugin, and `GT_ARTIFACT_DIR` defaults to `ml/artifacts/`.

### Java (plugin)

Build with Maven from the project root:

```
Lifecycle → install   (in IntelliJ Maven panel, or: mvn package)
```

Copy the built `GenerativeTerrain.jar` into `<server>/plugins/`.

---

## Data collection (in-game commands)

All collection commands export one CSV per chunk to `GT_DATA_DIR`. Run these in-game on a Minecraft server with the plugin loaded.

| Command | What it does |
|---|---|
| `/setdatapath <path>` | Set the server-relative directory for CSV exports |
| `/grabchunkdata` | Export the current chunk immediately |
| `/grabchunkarea [radius]` | Export every chunk in a square radius (one CSV each, async) |
| `/grabbiome <biome> [count] [searchRadius]` | Spiral outward and collect chunks of a specific biome |
| `/grabblock <block> [count] [searchRadius] [minPerChunk]` | Spiral outward and collect chunks containing a block |
| `/generateterrain [chunkX chunkZ] [style <name\|id>]` | Generate terrain with the loaded ONNX model (base style by default) |

**Biome-targeted collection** is important because the dataset is heavily imbalanced. Common biomes (FOREST, PLAINS, DARK_FOREST) can accumulate thousands of chunks naturally. Rare biomes (TAIGA, SAVANNA, FLOWER_FOREST) may appear only once or twice. Use `/grabbiome` to actively over-sample them:

```
/grabbiome taiga 20
/grabbiome savanna 20
/grabbiome flower_forest 20
/grabbiome desert 20
```

The sampler (`config.balance_biomes = True`) applies inverse-frequency weights at training time so rare biomes aren't drowned out even if they have fewer files — but more data is always better.

---

## Training

Open `ml/notebooks/GenerativeTerrain.ipynb` and run top to bottom. The notebook:

1. Loads all chunk CSVs into memory as `[16, 384, 16]` int8 voxel grids
2. Trains the deterministic **baseline** (`BaselineVoxelNet`) — layered terrain, no variety; confirms the pipeline is correct
3. Trains the **ConditionalTerrainVAE** — spatial latent, 3-level U-Net, heightmap stage, style conditioning; with EMA weights, mixed precision, warmup+cosine LR, and biome dropout
4. Evaluates both: per-group accuracy, vertical block-by-height profile (the terrain-plausibility check), VAE sample diversity
5. Freezes the model as a **versioned base checkpoint** (`artifacts/base_v1/`: weights + a manifest with the exact architecture and mappings) — the foundation fine-tuning builds on
6. Exports the VAE decoder to `terrain_vae_decoder.onnx` and writes the three JSON mappings

**For a fast CPU smoke test**, swap `cfg = Config()` for `cfg = Config.tiny()` at the top. Runs in ~1 minute, exercises everything, produces speckly terrain (it's undertrained by design).

**Run tests:**

```powershell
cd ml
python -m pytest tests/
```

### Key hyperparameters (`ml/gt_terrain/config.py`)

| Parameter | Default | Notes |
|---|---|---|
| `base_channels` | 32 | Width of 3D conv stacks. Scale up as dataset grows. |
| `latent_channels` | 8 | Channels per spatial latent cell. |
| `latent_grid` | `(4, 16, 4)` | Coarse latent resolution (x, y, z). Finer Y → better surface localisation. |
| `beta` | 1.0 | KL weight. KL is averaged per latent element so beta ~1 stays balanced with reconstruction. |
| `kl_anneal_epochs` | 10 | Epochs to ramp beta from 0 to `beta`. Lets the decoder learn to use `z` first. |
| `free_bits` | 0.02 | Nats per latent dim free from KL pressure. Prevents posterior collapse. |
| `heightmap_weight` | 1.0 | L1 supervision weight on the predicted surface heightmap. |
| `balance_biomes` | True | Biome-balanced sampler (weight ~ 1/biome_count). |
| `biome_dropout` | 0.05 | Fraction of chunks relabelled UNKNOWN so that row learns generic terrain. |
| `style_dim` / `max_custom_styles` | 64 / 8 | Style vector width; reserved fine-tuning slots. |
| `style_noise` | 0.1 | Style jitter during base training (keeps the style space fine-tunable). |
| `ema_decay` / `amp` | 0.999 / True | EMA weights are validated + exported; mixed precision on CUDA. |
| `clean_terrain` | True | Apply morphological cleanup to generated grids. |
| `ore_scatter` | True | Scatter ores procedurally. |
| `epochs` | 200 | With early stopping (`patience=20`), warmup+cosine LR. |

---

## Fine-tuning on custom terrain (transfer learning)

This is the headline feature: take the pretrained base and teach it *your* world's terrain as a new **style**, without retraining and without touching the base weights (a unit test asserts they come out bit-identical).

1. On your server, collect chunks of the terrain you want: `/grabchunkarea 5` (aim for 50+ chunks).
2. Open `ml/notebooks/FineTune.ipynb`, point it at the base checkpoint (`artifacts/base_v1/`), your CSV folder, and a style name.
3. Run it. `train.fine_tune()` rebuilds the exact base architecture from the checkpoint manifest, maps your data with the base's block grouping and biome encoder (unseen biomes → the trained UNKNOWN row), and trains only the new style vector + the FiLM projections — **~84k of the 4.6M parameters (1.8%)**. Minutes, not hours.
4. Export. One ONNX now carries both styles; in game:

```
/generateterrain                  ← vanilla base style
/generateterrain style myworld    ← your terrain's style
```

`config.ft_unfreeze` scales the adaptation: `"style"` (tiny), `"film"` (default), `"decoder"` (full decoder, for 100+ chunk datasets). Up to 8 custom styles fit in one model; re-using a name retrains that slot. To share, ship someone your `base_v1/` folder — that's all they need to fine-tune their own styles.

---

## Deploying the model in-game

After training (or fine-tuning), copy four files into `<server>/plugins/GenerativeTerrain/`:

```
ml/artifacts/terrain_vae_decoder.onnx
ml/artifacts/block_group_mapping.json
ml/artifacts/biome_id_mapping.json
ml/artifacts/style_mapping.json
```

Restart the server (or reload the plugin), then run `/generateterrain` while standing in a chunk. The plugin will:

1. Sample `z ~ N(0, I)` with the shape read from the ONNX graph
2. Look up the chunk's biome ID from `biome_id_mapping.json` (unknown biomes → the UNKNOWN row)
3. Resolve the requested style from `style_mapping.json` (default: base)
4. Run the decoder in a single ONNX inference call → `[27, 16, 384, 16]` logits
5. Argmax over the 27 block groups per voxel
6. Apply `TerrainPostProcessor` (cleanup + ore scatter)
7. Expand each block group to a specific Minecraft block using `block_group_mapping.json` and context rules (e.g. logs vs leaves by neighbours)
8. Place all blocks into the world

---

## Block groups

The model predicts one of **27 block groups** rather than individual block IDs. This keeps the output space tractable while preserving the distinctions that matter for terrain:

`AIR`, `STONE`, `DEEPSLATE`, `DIRT`, `GRASS`, `SAND`, `GRAVEL`, `CLAY`, `WATER`, `LAVA`, `BEDROCK`, `COAL_ORE`, `IRON_ORE`, `COPPER_ORE`, `GOLD_ORE`, `REDSTONE_ORE`, `LAPIS_ORE`, `DIAMOND_ORE`, `EMERALD_ORE`, `LOG`, `LEAVES`, `ICE`, `SNOW`, `MUSHROOM`, `NETHERRACK`, `SOUL_SAND`, `OTHER`

The mapping from group to specific blocks (and the logic for context-sensitive choices like dirt vs grass) is in `blocks.py` (Python) and reflected in the Java plugin. Both read the same `block_group_mapping.json`, so they stay in sync.

---

## Current state and roadmap

**Working now:**
- Clean, leakage-free data pipeline
- Spatial-latent VAE with 3-level 3D U-Net decoder, heightmap stage, and FiLM style conditioning
- **Transfer learning**: versioned base checkpoint + `fine_tune()` + `FineTune.ipynb` — custom terrain styles from a folder of CSVs, base weights provably untouched
- EMA, mixed precision, warmup+cosine LR, biome-balanced sampling, biome dropout
- Morphological post-processing + procedural ore scatter (Python + Java)
- Single ONNX forward pass from latent noise to placed blocks; style selectable in game

**Known limitations:**
- **Data is the bottleneck.** The VAE is data-hungry. A small dataset produces speckling and biome homogeneity. Collect aggressively with `/grabchunkarea` and the targeted commands.
- **Per-chunk independence.** Chunks are generated in isolation, so borders between adjacent generated chunks won't align. This is the next structural lever.
- **Rare-biome quality.** Biomes with fewer than ~20 chunks will be undertrained regardless of sampler weights. Bias collection toward rare biomes early.

**Planned next:**
- Cross-chunk edge conditioning for seamless multi-chunk generation (condition on already-generated neighbour edges — legitimate, since by then they exist).
- Publish a trained `base_v1` checkpoint so others can fine-tune without collecting a base dataset.

---

## License

MIT
