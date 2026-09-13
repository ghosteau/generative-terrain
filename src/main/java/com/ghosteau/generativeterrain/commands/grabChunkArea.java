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
 * {@code /grabchunkarea [radius]} -- bulk-exports every chunk in a square radius
 * (in chunks) around the player, one CSV per chunk. The fast way to build a
 * dataset instead of standing in each chunk and running {@code /grabchunkdata}.
 *
 * <p>For targeted collection of under-represented data, see {@link grabBiome}
 * and {@link grabBlock}.
 */
public class grabChunkArea implements CommandExecutor
{
    private final JavaPlugin plugin;

    private static final int MAX_RADIUS = 8;     // radius 8 == 17x17 = 289 chunks
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
            try { radius = Integer.parseInt(args[0]); }
            catch (NumberFormatException e) { player.sendMessage(ChatColor.YELLOW + "Invalid radius. Using " + DEFAULT_RADIUS + "."); }
        }
        radius = Math.max(0, Math.min(radius, MAX_RADIUS));

        final World world = player.getWorld();
        final Chunk center = world.getChunkAt(player.getLocation());
        List<int[]> coords = ChunkSearchTask.ringOrder(center.getX(), center.getZ(), radius);

        player.sendMessage(ChatColor.GREEN + "Exporting " + coords.size() + " chunks (radius " + radius + ")...");
        new ChunkSearchTask(plugin, player, world, coords, c -> true,
                ChunkSearchTask.prefixFrom(basePath), "", Integer.MAX_VALUE)
                .runTaskTimer(plugin, 1L, 1L);
        return true;
    }
}
