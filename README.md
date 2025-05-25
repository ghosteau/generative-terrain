# GenerativeTerrain — A Minecraft Plugin Project

This project explores the use of machine learning models to generate terrain within the game Minecraft, using both convolutional and transformer-based architectures.

## Overview

Two model types have been developed so far:

- CNN-based model — A 3D convolutional neural network trained on structured block data.
- Transformer-based model — A transformer architecture designed to learn spatial relationships across block positions.
- You can choose which model you want by simply modifying the private final variable "model" in `modelGenerateTerrain.java`

Both models are trained using PyTorch and exported to ONNX format for integration into a Java-based Minecraft plugin.

## Dependencies

All required plugin dependencies are located in the `plugins_server` folder and are meant to be used with a local PaperMC server.

- Your model `.onnx` and `.json` files should be placed in:
  `C:\Users\path_to_server\Server\plugins\GenerativeTerrain`

- For Java integration, use Maven to install dependencies such as the ONNX Runtime.
- The plugin requires JSON mappings to decode ONNX model outputs into Minecraft block structures. These are provided in the repo.
- Make sure in your `C:\Users\path_to_server\Server\plugins\` directory that there is a `GenerativeTerrain.jar` file (NOT the directory I mentioned earlier) which should be generated via Maven when you run the Maven install process on the project from you Java IDE.

## Model Training

- Training notebooks are provided in this repository.
- Data was extracted directly from Minecraft using a custom `/grabchunkdata` and `/setdatapath`command (also included in the plugin if you want to use it on your own!).
- A full dataset will be uploaded to Kaggle and/or Hugging Face along with the models when the project is finalized.

## Commands:
- You can set your data path (relative to your server file) using: `/setdatapath`
- After you set the data path, you can grab the current chunk you are on using `/grabchunkdata` which will extract a `.csv` file to your given directory
- If you want to generate terrain for the chunk you are on, use `/modelgenerateterrain`

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
