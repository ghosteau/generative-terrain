# GenerativeTerrain — A Minecraft Plugin Project

This project explores the use of machine learning models to generate terrain within the game Minecraft, using both convolutional and transformer-based architectures.

## Overview

Two model types have been developed so far:

- CNN-based model — A 3D convolutional neural network trained on structured block data.
- Transformer-based model — A transformer architecture designed to learn spatial relationships across block positions.

Both models are trained using PyTorch and exported to ONNX format for integration into a Java-based Minecraft plugin.

## Dependencies

All required plugin dependencies are located in the `plugins_server` folder and are meant to be used with a local PaperMC server.

- Your model `.onnx` and `.json` files should be placed in:
  C:\Users\path_to_server\Server\plugins\GenerativeTerrain

- For Java integration, use Maven to install dependencies such as the ONNX Runtime.
- The plugin requires JSON mappings to decode ONNX model outputs into Minecraft block structures. These are provided in the repo.

## Model Training

- Training notebooks are provided in this repository.
- Data was extracted directly from Minecraft using a custom `/extractdata` command (also included in the plugin).
- A full dataset will be uploaded to Kaggle when the project is finalized.

## Data Features

The dataset used for training includes the following features per block:

```python
[
    "X position",
    "Y position",
    "Z position",
    "Chunk Biome",
    "Block Biome",
    "Block ID",
    "Is Surface Block",
    "Light Level",
    "Block to the Left",
    "Block to the Right",
    "Block Below",
    "Block Above",
    "Block in Front",
    "Block Behind"
]
```
