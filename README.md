# GenerativeTerrain

A Minecraft PaperMC plugin and PyTorch training pipeline that uses a **conditional VAE** to generate terrain chunk-by-chunk from a sampled latent noise vector and a biome ID.

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

### Architecture

```
                    Training                              Generation
                       ↓                                     ↓
Real chunk (16×384×16) → Encoder → μ, logσ²     z ~ N(0, I) (spatial grid)
                                      ↓                      ↓
                                 Reparameterize → z ──→ Decoder
                                                          ↓       ↓
                                             Stage 1: HeightmapHead
                                             (per-column surface in [0,1])
                                                          ↓
                                             signed-distance-to-surface channel
                                                          ↓
                                             Stage 2: 3D U-Net fills volume
                                                          ↓
                                             Logits [27, 16, 384, 16]
```

**Spatial latent** — `z` is a coarse 3D grid `(8, 4, 16, 4)` rather than a single broadcast vector. A broadcast vector can only produce the average column per biome (flat terrain, no hills, no caves). The spatial grid lets each region of the chunk vary independently.

**Heightmap stage** — before filling every voxel the decoder predicts a per-column surface height from `z` + biome. That surface feeds a signed-distance channel (negative below ground, positive above) to the U-Net. This removes the up/down ambiguity that caused the model to carve out air in hilly biomes.

**3D U-Net decoder** — two down/up levels with skip connections give the model a chunk-scale receptive field so it can form coherent hills, valleys, and cave systems rather than predicting each voxel from a tiny local window.

**Post-processing** — the model predicts terrain shape; two deterministic steps are applied afterwards (matching how vanilla Minecraft works):
- *Morphological cleanup*: remove single floating blocks and fill single-voxel pinholes. Keeps real caves; removes speckle.
- *Procedural ore scatter*: place coal/iron/diamond/etc. into stone and deepslate at vanilla-ish Y ranges and probabilities. Learning rare scattered classes from a VAE is unreliable; doing it procedurally is controllable and always works.

Both steps are mirrored exactly in the Java plugin (`TerrainPostProcessor.java`).

**ONNX export** — only the decoder is exported: `(z, biome_id) → logits`. The encoder is training-only. The Java plugin samples `z ~ N(0, I)`, reads the z shape directly from the ONNX graph, and runs a single inference call.

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
│   │   ├── models.py       BaselineVoxelNet + ConditionalTerrainVAE
│   │   ├── train.py        Training loops (KL annealing, free bits, class weights)
│   │   ├── evaluate.py     Accuracy, vertical profiles, sample diversity
│   │   ├── generate.py     Sample chunks from the trained VAE
│   │   ├── postprocess.py  Morphological clean + procedural ore scatter
│   │   └── export.py       ONNX export + JSON mapping files
│   ├── notebooks/
│   │   ├── GenerativeTerrain.ipynb    The single driver notebook
│   │   └── legacy/                   Six superseded notebooks kept for reference
│   ├── tests/
│   │   └── test_data.py    pytest suite including a no-leakage invariant
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
| `/modelgenerateterrain` | Generate terrain for the current chunk using the loaded ONNX model |

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

1. Loads all chunk CSVs into memory as `[16, 384, 16]` int16 voxel grids
2. Trains the deterministic **baseline** (`BaselineVoxelNet`) — layered terrain, no variety; confirms the pipeline is correct
3. Trains the **ConditionalTerrainVAE** — generative, spatial latent, U-Net decoder, heightmap stage
4. Evaluates both: per-group accuracy, vertical block-by-height profile (the terrain-plausibility check), VAE sample diversity
5. Exports the VAE decoder to `terrain_vae_decoder.onnx` and writes `block_group_mapping.json` + `biome_id_mapping.json`

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
| `clean_terrain` | True | Apply morphological cleanup to generated grids. |
| `ore_scatter` | True | Scatter ores procedurally. |
| `epochs` | 200 | With early stopping (`patience=20`). |

---

## Deploying the model in-game

After training, copy three files into `<server>/plugins/GenerativeTerrain/`:

```
ml/artifacts/terrain_vae_decoder.onnx
ml/artifacts/block_group_mapping.json
ml/artifacts/biome_id_mapping.json
```

Restart the server (or reload the plugin), then run `/modelgenerateterrain` while standing in a chunk. The plugin will:

1. Sample `z ~ N(0, I)` with the shape read from the ONNX graph
2. Look up the chunk's biome ID from `biome_id_mapping.json`
3. Run the decoder in a single ONNX inference call → `[27, 16, 384, 16]` logits
4. Argmax over the 27 block groups per voxel
5. Apply `TerrainPostProcessor` (cleanup + ore scatter)
6. Expand each block group to a specific Minecraft block using `block_group_mapping.json` and context rules (e.g. top stone → grass if near surface)
7. Place all blocks into the world

---

## Block groups

The model predicts one of **27 block groups** rather than individual block IDs. This keeps the output space tractable while preserving the distinctions that matter for terrain:

`AIR`, `STONE`, `DEEPSLATE`, `DIRT`, `GRASS`, `SAND`, `GRAVEL`, `CLAY`, `WATER`, `LAVA`, `BEDROCK`, `COAL_ORE`, `IRON_ORE`, `COPPER_ORE`, `GOLD_ORE`, `REDSTONE_ORE`, `LAPIS_ORE`, `DIAMOND_ORE`, `EMERALD_ORE`, `LOG`, `LEAVES`, `ICE`, `SNOW`, `MUSHROOM`, `NETHERRACK`, `SOUL_SAND`, `OTHER`

The mapping from group to specific blocks (and the logic for context-sensitive choices like dirt vs grass) is in `blocks.py` (Python) and reflected in the Java plugin. Both read the same `block_group_mapping.json`, so they stay in sync.

---

## Current state and roadmap

**Working now:**
- Clean, leakage-free data pipeline
- Spatial-latent VAE with 3D U-Net decoder and heightmap stage
- Biome-balanced training sampler + targeted collection commands
- Morphological post-processing + procedural ore scatter (Python + Java)
- Single ONNX forward pass from latent noise to placed blocks

**Known limitations:**
- **Data is the bottleneck.** The VAE is data-hungry. A small dataset produces speckling and biome homogeneity. Collect aggressively with `/grabchunkarea` and the targeted commands.
- **Per-chunk independence.** Chunks are generated in isolation, so borders between adjacent generated chunks won't align. This is the next structural lever after data grows.
- **Rare-biome quality.** Biomes with fewer than ~20 chunks will be undertrained regardless of sampler weights. Bias collection toward rare biomes early.

**Planned next:**
- Fine-tuning foundation: lock a versioned base checkpoint, add `fine_tune()` entry point, extensible style embedding (reserve trainable custom slots alongside the biome embedding). Goal: server owners train on their own terrain → a generator with their world's style, without retraining from scratch.

---

## License

MIT
