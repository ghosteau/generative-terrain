package com.ghosteau.generativeterrain.commands;

import org.bukkit.ChatColor;
import org.bukkit.Chunk;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.List;

/**
 * {@code /grabblock <block> [count] [searchRadius] [minPerChunk]} -- targeted
 * collection of chunks that contain a specific block.
 *
 * <p>Spirals outward and exports the first {@code count} chunks that contain at
 * least {@code minPerChunk} of {@code <block>}. Useful for over-sampling rare
 * non-ore features the model seldom sees (e.g. lots of WATER, or JUNGLE_LOG).
 *
 * <p>Block is matched with {@link Material#matchMaterial}, e.g.
 * {@code /grabblock water 20 16 64}.
 */
public class grabBlock implements CommandExecutor
{
    private final JavaPlugin plugin;

    private static final int DEFAULT_COUNT = 16;
    private static final int DEFAULT_RADIUS = 12;
    private static final int MAX_RADIUS = 32;
    private static final int DEFAULT_MIN_PER_CHUNK = 1;

    public grabBlock(JavaPlugin plugin)
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

        if (args.length < 1)
        {
            player.sendMessage(ChatColor.RED + "Usage: /grabblock <block> [count] [searchRadius] [minPerChunk]");
            return true;
        }

        final Material mat = Material.matchMaterial(args[0]);
        if (mat == null || !mat.isBlock())
        {
            player.sendMessage(ChatColor.RED + "Unknown block: " + args[0]);
            return true;
        }
        int count = parseOr(args, 1, DEFAULT_COUNT);
        int radius = Math.min(parseOr(args, 2, DEFAULT_RADIUS), MAX_RADIUS);
        final int minPerChunk = Math.max(1, parseOr(args, 3, DEFAULT_MIN_PER_CHUNK));

        final World world = player.getWorld();
        final Chunk center = world.getChunkAt(player.getLocation());
        List<int[]> coords = ChunkSearchTask.ringOrder(center.getX(), center.getZ(), radius);

        player.sendMessage(ChatColor.GREEN + "Searching up to " + coords.size()
                + " chunks for " + mat + " (>=" + minPerChunk + " per chunk, want " + count + ")...");
        new ChunkSearchTask(plugin, player, world, coords,
                chunk -> ChunkSearchTask.chunkHasAtLeast(chunk, mat, minPerChunk),
                ChunkSearchTask.prefixFrom(basePath), mat.toString(), count)
                .runTaskTimer(plugin, 1L, 1L);
        return true;
    }

    private static int parseOr(String[] args, int i, int fallback)
    {
        if (args.length > i)
        {
            try { return Integer.parseInt(args[i]); }
            catch (NumberFormatException ignored) { }
        }
        return fallback;
    }
}
