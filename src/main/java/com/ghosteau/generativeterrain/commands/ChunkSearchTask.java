package com.ghosteau.generativeterrain.commands;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Chunk;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Spirals outward from a centre chunk, exporting chunks that match a predicate,
 * one chunk per tick (block reads on the main thread, file writes async).
 *
 * <p>Shared by {@link grabChunkArea} (match-all), {@link grabBiome} (match a
 * target biome) and {@link grabBlock} (match chunks containing a target block).
 * Searching one chunk per tick keeps the server responsive even though scanning
 * unexplored area forces chunk generation.
 */
public class ChunkSearchTask extends BukkitRunnable
{
    private final JavaPlugin plugin;
    private final Player player;
    private final World world;
    private final List<int[]> coords;
    private final Predicate<Chunk> match;
    private final String prefix;
    private final String tag;
    private final int maxMatches;

    private int index = 0;
    private int matched = 0;

    public ChunkSearchTask(JavaPlugin plugin, Player player, World world, List<int[]> coords,
                           Predicate<Chunk> match, String prefix, String tag, int maxMatches)
    {
        this.plugin = plugin;
        this.player = player;
        this.world = world;
        this.coords = coords;
        this.match = match;
        this.prefix = prefix;
        this.tag = tag;
        this.maxMatches = maxMatches;
    }

    @Override
    public void run()
    {
        if (index >= coords.size() || matched >= maxMatches)
        {
            player.sendMessage(ChatColor.GREEN + "" + ChatColor.BOLD + "[!] Done. Exported "
                    + matched + " chunk CSV(s) (scanned " + index + ").");
            cancel();
            return;
        }

        int[] c = coords.get(index++);
        try
        {
            Chunk chunk = world.getChunkAt(c[0], c[1]); // loads / generates if needed
            if (match.test(chunk))
            {
                final String csv = ChunkDataExtractor.buildCsv(chunk);
                final File out = new File(prefix + (tag.isEmpty() ? "" : "_" + tag)
                        + "_" + c[0] + "_" + c[1] + ".csv");
                matched++;
                Bukkit.getScheduler().runTaskAsynchronously(plugin, () ->
                {
                    try { ChunkDataExtractor.writeCsv(out, csv); }
                    catch (Exception e) { plugin.getLogger().warning("Write failed " + out.getName() + ": " + e.getMessage()); }
                });
            }
        }
        catch (Exception e)
        {
            plugin.getLogger().warning("Scan failed at " + c[0] + "," + c[1] + ": " + e.getMessage());
        }

        if (index % 25 == 0)
        {
            player.sendMessage(ChatColor.GRAY + "Scanned " + index + "/" + coords.size()
                    + " (" + matched + "/" + maxMatches + " collected)");
        }
    }

    // --- shared helpers ----------------------------------------------------

    /** Turn a configured ".csv" data path into a file-name prefix. */
    public static String prefixFrom(String basePath)
    {
        return basePath.toLowerCase().endsWith(".csv")
                ? basePath.substring(0, basePath.length() - 4)
                : basePath;
    }

    /** Chunk coordinates within {@code radius}, ordered nearest-first (by ring). */
    public static List<int[]> ringOrder(int cx, int cz, int radius)
    {
        List<int[]> out = new ArrayList<>();
        for (int r = 0; r <= radius; r++)
        {
            if (r == 0) { out.add(new int[]{cx, cz}); continue; }
            for (int dx = -r; dx <= r; dx++)
                for (int dz = -r; dz <= r; dz++)
                    if (Math.max(Math.abs(dx), Math.abs(dz)) == r)
                        out.add(new int[]{cx + dx, cz + dz});
        }
        return out;
    }

    /** Biome name at the centre of a chunk (matches the ChunkBiome export convention). */
    public static String chunkBiomeName(Chunk chunk)
    {
        int x = chunk.getX() * 16 + 8, z = chunk.getZ() * 16 + 8;
        return chunk.getWorld().getBiome(x, 64, z).toString();
    }

    /** True once a chunk contains at least {@code minCount} of {@code mat} (early-exits). */
    public static boolean chunkHasAtLeast(Chunk chunk, Material mat, int minCount)
    {
        int count = 0;
        for (int x = 0; x < 16; x++)
            for (int y = ChunkDataExtractor.MIN_Y; y <= ChunkDataExtractor.MAX_Y; y++)
                for (int z = 0; z < 16; z++)
                    if (chunk.getBlock(x, y, z).getType() == mat && ++count >= minCount)
                        return true;
        return false;
    }
}
