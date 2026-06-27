"""
Block grouping -- the single source of truth.

Minecraft has ~180 distinct block types in our data, most of them rare. We
collapse them into 27 semantic *groups* (e.g. all six log/leaf pairs, ores with
their deepslate variants, the stone family). This:

* turns a long-tailed 180-way problem into a tractable 27-way one;
* lets the Java plugin re-expand a group into a concrete block using local
  context (deepslate vs stone ores by height, log vs leaf by neighbours).

The grouping previously lived *twice* -- hard-coded in ``GTModelingV3.ipynb`` and
again as a Java array in ``modelGenerateTerrain.java``. Here it lives once, is
loaded from ``block_grouping_config.json`` when present (so the historical group
ordering is preserved), and is exported back out for Java to read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Canonical grouping. Order defines the integer class ids (AIR == 0, ...).
# Kept byte-for-byte consistent with the historical block_grouping_config.json
# so previously-exported models/mappings stay comparable.
BLOCK_GROUPS: dict[str, list[str]] = {
    "AIR": ["AIR", "CAVE_AIR", "VOID_AIR"],
    "STONE": ["STONE", "ANDESITE", "DIORITE", "GRANITE", "TUFF"],
    "DEEPSLATE": ["DEEPSLATE"],
    "DIRT": ["DIRT", "COARSE_DIRT", "ROOTED_DIRT"],
    "GRASS": ["GRASS_BLOCK", "PODZOL"],
    "SAND": ["SAND", "RED_SAND"],
    "GRAVEL": ["GRAVEL"],
    "CLAY": ["CLAY"],
    "WATER": ["WATER"],
    "LAVA": ["LAVA"],
    "BEDROCK": ["BEDROCK"],
    "COAL_ORE": ["COAL_ORE", "DEEPSLATE_COAL_ORE"],
    "IRON_ORE": ["IRON_ORE", "DEEPSLATE_IRON_ORE"],
    "COPPER_ORE": ["COPPER_ORE", "DEEPSLATE_COPPER_ORE"],
    "GOLD_ORE": ["GOLD_ORE", "DEEPSLATE_GOLD_ORE"],
    "REDSTONE_ORE": ["REDSTONE_ORE", "DEEPSLATE_REDSTONE_ORE"],
    "LAPIS_ORE": ["LAPIS_ORE", "DEEPSLATE_LAPIS_ORE"],
    "DIAMOND_ORE": ["DIAMOND_ORE", "DEEPSLATE_DIAMOND_ORE"],
    "EMERALD_ORE": ["EMERALD_ORE", "DEEPSLATE_EMERALD_ORE"],
    "OAK_WOOD": ["OAK_LOG", "OAK_LEAVES"],
    "SPRUCE_WOOD": ["SPRUCE_LOG", "SPRUCE_LEAVES"],
    "BIRCH_WOOD": ["BIRCH_LOG", "BIRCH_LEAVES"],
    "JUNGLE_WOOD": ["JUNGLE_LOG", "JUNGLE_LEAVES"],
    "ACACIA_WOOD": ["ACACIA_LOG", "ACACIA_LEAVES"],
    "DARK_OAK_WOOD": ["DARK_OAK_LOG", "DARK_OAK_LEAVES"],
    "VEGETATION": [
        "SHORT_GRASS", "TALL_GRASS", "SEAGRASS", "TALL_SEAGRASS",
        "KELP", "KELP_PLANT", "LILY_PAD",
    ],
    "MISC": [
        "GLOW_LICHEN", "MOSS_BLOCK", "MOSS_CARPET",
        "SCULK", "SCULK_VEIN", "POINTED_DRIPSTONE",
    ],
}

# The id every unmapped / unknown block falls back to. AIR is the safe default:
# an unknown rare block becoming air is far less jarring than becoming stone.
FALLBACK_GROUP = "AIR"


class BlockGrouping:
    """Bidirectional block <-> group lookups, plus fast array mapping."""

    def __init__(self, groups: dict[str, list[str]] | None = None):
        self.groups = groups if groups is not None else BLOCK_GROUPS
        self.group_names: list[str] = list(self.groups.keys())
        self.group_to_idx: dict[str, int] = {g: i for i, g in enumerate(self.group_names)}
        self.idx_to_group: dict[int, str] = {i: g for g, i in self.group_to_idx.items()}

        # Concrete block name -> group id
        self.block_to_idx: dict[str, int] = {}
        for group_name, members in self.groups.items():
            gid = self.group_to_idx[group_name]
            for block in members:
                self.block_to_idx[block] = gid

        self.fallback_idx = self.group_to_idx[FALLBACK_GROUP]

    @property
    def num_classes(self) -> int:
        return len(self.group_names)

    def block_id(self, block_name: str) -> int:
        """Map one block name to its group id (fallback for unknowns)."""
        return self.block_to_idx.get(block_name, self.fallback_idx)

    def map_block_array(self, block_names: np.ndarray) -> np.ndarray:
        """Vectorised map of an array of block-name strings to group ids.

        This replaces the old per-row ``LabelEncoder.inverse_transform`` loop,
        which dominated preprocessing time.
        """
        # np.vectorize builds a small C loop over the dict lookup; for ~100k
        # entries per chunk this is plenty fast and keeps the code obvious.
        lookup = np.vectorize(self.block_id, otypes=[np.int64])
        return lookup(block_names)

    # --- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "BlockGrouping":
        """Load a grouping from a ``block_grouping_config.json`` style file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(groups=data["block_groups"])

    def to_config_dict(self) -> dict:
        return {
            "block_groups": self.groups,
            "group_to_idx": self.group_to_idx,
            "num_classes": self.num_classes,
        }

    def save_group_mapping(self, path: str | Path) -> None:
        """Write ``{id: GROUP_NAME}`` -- the file the Java plugin decodes with."""
        mapping = {str(i): name for i, name in self.idx_to_group.items()}
        Path(path).write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def load_grouping(config_json: str | Path | None = None) -> BlockGrouping:
    """Convenience: load from JSON if it exists, else use the built-in default."""
    if config_json is not None and Path(config_json).exists():
        return BlockGrouping.load(config_json)
    return BlockGrouping()
