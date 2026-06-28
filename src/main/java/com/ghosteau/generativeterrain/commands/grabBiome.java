package com.ghosteau.generativeterrain.commands;

import org.bukkit.ChatColor;
import org.bukkit.Chunk;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.List;

/**
 * {@code /grabbiome <biome> [count] [searchRadius]} -- targeted collection of a
 * specific (usually rare) biome.
 *
 * <p>Spirals outward from the player up to {@code searchRadius} chunks and
 * exports the first {@code count} chunks whose biome matches {@code <biome>}.
 * Use this to balance a FOREST/PLAINS-heavy dataset with rare biomes like
 * TAIGA or SAVANNA without hunting for them by hand.
 *
 * <p>Biome is matched by name (case-insensitive), e.g. {@code /grabbiome taiga 20 16}.
 */
public class grabBiome implements CommandExecutor
{
    private final JavaPlugin plugin;

    private static final int DEFAULT_COUNT = 16;
    private static final int DEFAULT_RADIUS = 12;
    private static final int MAX_RADIUS = 32;

    public grabBiome(JavaPlugin plugin)
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
            player.sendMessage(ChatColor.RED + "Usage: /grabbiome <biome> [count] [searchRadius]");
            return true;
        }

        final String target = args[0].toUpperCase();
        int count = parseOr(args, 1, DEFAULT_COUNT);
        int radius = Math.min(parseOr(args, 2, DEFAULT_RADIUS), MAX_RADIUS);

        final World world = player.getWorld();
        final Chunk center = world.getChunkAt(player.getLocation());
        List<int[]> coords = ChunkSearchTask.ringOrder(center.getX(), center.getZ(), radius);

        player.sendMessage(ChatColor.GREEN + "Searching up to " + coords.size()
                + " chunks for biome " + target + " (want " + count + ")...");
        new ChunkSearchTask(plugin, player, world, coords,
                chunk -> ChunkSearchTask.chunkBiomeName(chunk).equalsIgnoreCase(target),
                ChunkSearchTask.prefixFrom(basePath), target, count)
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
