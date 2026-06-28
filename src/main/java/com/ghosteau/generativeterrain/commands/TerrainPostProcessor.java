package com.ghosteau.generativeterrain.commands;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Deterministic post-processing of the model's group grid, mirroring the Python
 * {@code gt_terrain.postprocess} module so in-game terrain matches the notebook.
 *
 * <ul>
 *   <li><b>clean</b> -- remove single floating blocks and fill single-voxel
 *       pinholes (tidies the fuzzy surface a generative model produces, without
 *       touching multi-voxel caves);</li>
 *   <li><b>scatter ores</b> -- sprinkle ore groups into stone/deepslate by depth
 *       and rarity (learning rare scattered blocks is unreliable; scattering them
 *       procedurally is controllable and always works).</li>
 * </ul>
 *
 * Operates on a {@code [X][Y][Z]} grid of group ids where Y is local
 * (0..height-1); world Y = local Y + minY.
 */
public final class TerrainPostProcessor
{
    // Ore -> {minWorldY, maxWorldY, per-voxel probability x1e6}. Mirrors ORE_SCATTER in postprocess.py.
    private static final Object[][] ORE_SCATTER = {
            {"COAL_ORE",        0, 256, 0.012},
            {"IRON_ORE",      -64, 256, 0.009},
            {"COPPER_ORE",    -16, 112, 0.007},
            {"REDSTONE_ORE",  -64,  15, 0.008},
            {"GOLD_ORE",      -64,  32, 0.0030},
            {"LAPIS_ORE",     -64,  64, 0.0025},
            {"DIAMOND_ORE",   -64,  16, 0.0018},
            {"EMERALD_ORE",   -16, 256, 0.0010},
    };

    private TerrainPostProcessor() { }

    /** Clean then scatter ores, in place. */
    public static void process(int[][][] grid, Map<Integer, String> groupDecoder,
                               int minY, double oreDensity, Random rng)
    {
        Map<String, Integer> id = new HashMap<>();
        for (Map.Entry<Integer, String> e : groupDecoder.entrySet()) id.put(e.getValue(), e.getKey());
        if (!id.containsKey("AIR") || !id.containsKey("STONE") || !id.containsKey("DEEPSLATE")) return;

        int air = id.get("AIR"), stone = id.get("STONE"), deepslate = id.get("DEEPSLATE");
        clean(grid, air, stone, deepslate, minY);
        scatterOres(grid, id, stone, deepslate, minY, oreDensity, rng);
    }

    private static void clean(int[][][] grid, int air, int stone, int deepslate, int minY)
    {
        int X = grid.length, Y = grid[0].length, Z = grid[0][0].length;

        // Pass 1: remove solid voxels with zero solid neighbours (floating specks).
        int[][][] orig = copy(grid);
        for (int x = 0; x < X; x++)
            for (int y = 0; y < Y; y++)
                for (int z = 0; z < Z; z++)
                    if (orig[x][y][z] != air && solidNeighbours(orig, x, y, z, air) == 0)
                        grid[x][y][z] = air;

        // Pass 2: fill air voxels fully enclosed by solid (1-voxel pinholes).
        orig = copy(grid);
        for (int x = 0; x < X; x++)
            for (int y = 0; y < Y; y++)
                for (int z = 0; z < Z; z++)
                    if (orig[x][y][z] == air && solidNeighbours(orig, x, y, z, air) == 6)
                        grid[x][y][z] = (y + minY < 0) ? deepslate : stone;
    }

    private static void scatterOres(int[][][] grid, Map<String, Integer> id,
                                    int stone, int deepslate, int minY, double density, Random rng)
    {
        int X = grid.length, Y = grid[0].length, Z = grid[0][0].length;
        for (int x = 0; x < X; x++)
        {
            for (int y = 0; y < Y; y++)
            {
                int worldY = y + minY;
                for (int z = 0; z < Z; z++)
                {
                    if (grid[x][y][z] != stone && grid[x][y][z] != deepslate) continue;
                    for (Object[] ore : ORE_SCATTER)
                    {
                        int yMin = (int) ore[1], yMax = (int) ore[2];
                        if (worldY < yMin || worldY > yMax) continue;
                        double p = ((double) ore[3]) * density;
                        if (rng.nextDouble() < p)
                        {
                            Integer gid = id.get((String) ore[0]);
                            if (gid != null) { grid[x][y][z] = gid; }
                            break;  // one ore per voxel
                        }
                    }
                }
            }
        }
    }

    /** Count of the 6 face-neighbours that are solid; out-of-bounds counts as solid. */
    private static int solidNeighbours(int[][][] g, int x, int y, int z, int air)
    {
        return isSolid(g, x - 1, y, z, air) + isSolid(g, x + 1, y, z, air)
             + isSolid(g, x, y - 1, z, air) + isSolid(g, x, y + 1, z, air)
             + isSolid(g, x, y, z - 1, air) + isSolid(g, x, y, z + 1, air);
    }

    private static int isSolid(int[][][] g, int x, int y, int z, int air)
    {
        if (x < 0 || y < 0 || z < 0 || x >= g.length || y >= g[0].length || z >= g[0][0].length)
            return 1;  // edge-replicate: treat out-of-bounds as solid
        return g[x][y][z] != air ? 1 : 0;
    }

    private static int[][][] copy(int[][][] g)
    {
        int X = g.length, Y = g[0].length, Z = g[0][0].length;
        int[][][] c = new int[X][Y][Z];
        for (int x = 0; x < X; x++)
            for (int y = 0; y < Y; y++)
                System.arraycopy(g[x][y], 0, c[x][y], 0, Z);
        return c;
    }
}
