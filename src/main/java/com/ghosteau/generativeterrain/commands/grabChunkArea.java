package com.ghosteau.generativeterrain.commands;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Chunk;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * {@code /grabchunkarea [radius]} -- bulk version of {@link grabChunkData}.
 *
 * <p>Exports every chunk in a square of the given radius (in chunks) around the
 * player, writing one CSV per chunk. This is the fast way to build a dataset
 * instead of standing in each chunk and running {@code /grabchunkdata}.
 *
 * <p>File names are derived from the {@code /setdatapath} value: the trailing
 * {@code .csv} is treated as a prefix, so {@code data/world.csv} with a chunk at
 * (12, -3) produces {@code data/world_12_-3.csv}. The training loader simply
 * reads every {@code *.csv} in the directory.
 *
 * <p>To avoid freezing the server, chunks are processed one per tick: block
 * reading happens on the main thread, file writing is offloaded to an async task.
 */
public class grabChunkArea implements CommandExecutor
{
    private final JavaPlugin plugin;

    /** Safety cap: radius 8 == a 17x17 = 289-chunk area. */
    private static final int MAX_RADIUS = 8;
    private static final int DEFAULT_RADIUS = 2;

    public grabChunkArea(JavaPlugin plugin)
    {
        this.plugin = plugin;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args)
    {
        if (!(sender instanceof Player))
        {
            sender.sendMessage(ChatColor.RED + "You must be in-game to execute this command.");
            return true;
        }

        Player player = (Player) sender;
        if (!player.hasPermission("generativeterrain.grabchunkdata"))
        {
            player.sendMessage(ChatColor.RED + "You don't have permission to use this command.");
            return true;
        }

        String basePath = setDataPath.getPath();
        if (basePath == null || basePath.isEmpty())
        {
            player.sendMessage(ChatColor.RED + "Warning: Please set a path first via /setDataPath");
            return true;
        }

        int radius = DEFAULT_RADIUS;
        if (args.length >= 1)
        {
            try
            {
                radius = Integer.parseInt(args[0]);
            }
            catch (NumberFormatException e)
            {
                player.sendMessage(ChatColor.YELLOW + "Invalid radius. Using " + DEFAULT_RADIUS + ".");
            }
        }
        radius = Math.max(0, Math.min(radius, MAX_RADIUS));

        // Derive a file-name prefix from the configured path (strip trailing .csv).
        String prefix = basePath.toLowerCase().endsWith(".csv")
                ? basePath.substring(0, basePath.length() - 4)
                : basePath;

        final World world = player.getWorld();
        final Chunk center = world.getChunkAt(player.getLocation());

        // Collect the chunk coordinates to export.
        final List<int[]> coords = new ArrayList<>();
        for (int dx = -radius; dx <= radius; dx++)
            for (int dz = -radius; dz <= radius; dz++)
                coords.add(new int[]{ center.getX() + dx, center.getZ() + dz });

        player.sendMessage(ChatColor.GREEN + "Exporting " + coords.size()
                + " chunks (radius " + radius + ")...");

        // One chunk per tick: read on the main thread, write to disk async.
        new BukkitRunnable()
        {
            int index = 0;

            @Override
            public void run()
            {
                if (index >= coords.size())
                {
                    player.sendMessage(ChatColor.GREEN + "" + ChatColor.BOLD
                            + "[!] Done. Exported " + coords.size() + " chunk CSVs.");
                    cancel();
                    return;
                }

                int[] c = coords.get(index++);
                try
                {
                    Chunk chunk = world.getChunkAt(c[0], c[1]); // loads the chunk if needed
                    String csv = ChunkDataExtractor.buildCsv(chunk);
                    File out = new File(prefix + "_" + c[0] + "_" + c[1] + ".csv");

                    Bukkit.getScheduler().runTaskAsynchronously(plugin, () ->
                    {
                        try
                        {
                            ChunkDataExtractor.writeCsv(out, csv);
                        }
                        catch (Exception e)
                        {
                            plugin.getLogger().warning("Failed to write " + out.getName() + ": " + e.getMessage());
                        }
                    });
                }
                catch (Exception e)
                {
                    plugin.getLogger().warning("Failed to read chunk " + c[0] + "," + c[1] + ": " + e.getMessage());
                }

                if (index % 10 == 0 || index == coords.size())
                {
                    player.sendMessage(ChatColor.GRAY + "Progress: " + index + "/" + coords.size());
                }
            }
        }.runTaskTimer(plugin, 1L, 1L);

        return true;
    }
}
