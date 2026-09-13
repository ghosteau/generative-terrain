package com.ghosteau.generativeterrain.commands;

import org.bukkit.Chunk;
import org.bukkit.Material;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

/**
 * Shared chunk -> CSV extraction used by both {@link grabChunkData} (single chunk)
 * and {@link grabChunkArea} (bulk). Keeping the feature schema in one place means
 * the single and bulk commands can never drift apart.
 *
 * <p>The CSV is one row per voxel over the full world height (Y -64..319), i.e.
 * 16 x 384 x 16 = 98,304 rows. {@link #buildCsv} reads blocks and MUST run on the
 * main server thread; {@link #writeCsv} only does file I/O and is safe to run async.
 */
public final class ChunkDataExtractor
{
    public static final int MIN_Y = -64;
    public static final int MAX_Y = 319;

    public static final String CSV_HEADER =
            "x,y,z,ChunkBiome,Biome,Block_ID,Is_Surface,Light_Level,"
            + "Block_to_Left,Block_to_Right,Block_Below,Block_Above,Block_in_Front,Block_Behind";

    private ChunkDataExtractor() { }

    /**
     * Build the full CSV text for one chunk. Must be called on the main thread
     * (Bukkit block access is not thread-safe).
     */
    public static String buildCsv(Chunk chunk)
    {
        // Chunk biome label: biome at the surface of the chunk's centre column.
        int centerX = 8, centerZ = 8, surfaceY = MAX_Y;
        while (surfaceY > MIN_Y && chunk.getBlock(centerX, surfaceY, centerZ).getType() == Material.AIR)
        {
            surfaceY--;
        }
        String chunkBiome = chunk.getBlock(centerX, surfaceY, centerZ).getBiome().toString();

        StringBuilder sb = new StringBuilder(CSV_HEADER.length() + 98_304 * 64);
        sb.append(CSV_HEADER).append('\n');

        for (int x = 0; x < 16; x++)
        {
            for (int y = MIN_Y; y <= MAX_Y; y++)
            {
                for (int z = 0; z < 16; z++)
                {
                    Block b = chunk.getBlock(x, y, z);
                    boolean isSurface = b.getRelative(BlockFace.UP).getType() == Material.AIR;

                    sb.append(x).append(',').append(y).append(',').append(z).append(',')
                      .append(chunkBiome).append(',')
                      .append(b.getBiome()).append(',')
                      .append(b.getType()).append(',')
                      .append(isSurface).append(',')
                      .append((double) b.getLightLevel()).append(',')
                      .append(b.getRelative(BlockFace.WEST).getType()).append(',')   // Block_to_Left
                      .append(b.getRelative(BlockFace.EAST).getType()).append(',')   // Block_to_Right
                      .append(b.getRelative(BlockFace.DOWN).getType()).append(',')   // Block_Below
                      .append(b.getRelative(BlockFace.UP).getType()).append(',')     // Block_Above
                      .append(b.getRelative(BlockFace.NORTH).getType()).append(',')  // Block_in_Front
                      .append(b.getRelative(BlockFace.SOUTH).getType())              // Block_Behind
                      .append('\n');
                }
            }
        }
        return sb.toString();
    }

    /** Write CSV text to disk, creating parent directories. Safe to call off-thread. */
    public static void writeCsv(File outFile, String content) throws IOException
    {
        File parent = outFile.getParentFile();
        if (parent != null && !parent.exists())
        {
            parent.mkdirs();
        }
        try (BufferedWriter w = new BufferedWriter(new FileWriter(outFile)))
        {
            w.write(content);
        }
    }

    /** Convenience: build + write one chunk (main thread). */
    public static void writeChunkCsv(Chunk chunk, File outFile) throws IOException
    {
        writeCsv(outFile, buildCsv(chunk));
    }
}
