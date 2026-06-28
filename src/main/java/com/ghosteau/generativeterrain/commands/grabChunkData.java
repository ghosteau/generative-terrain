package com.ghosteau.generativeterrain.commands;

import org.bukkit.Chunk;
import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

import java.io.File;

/**
 * {@code /grabchunkdata} -- exports the player's current chunk to a CSV used to
 * train the terrain model (one row per voxel; see {@link ChunkDataExtractor}).
 *
 * <p>For collecting many chunks at once, use {@link grabChunkArea}.
 *
 * <p>Note: the training pipeline uses only {@code x, y, z, ChunkBiome, Block_ID};
 * the neighbour / light / surface columns are exported for analysis but are not
 * model inputs (using them would leak the terrain being generated).
 */
public class grabChunkData implements CommandExecutor
{
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

        if (setDataPath.getPath() == null || setDataPath.getPath().isEmpty())
        {
            player.sendMessage(ChatColor.RED + "Warning: Please set a path before using this command via /setDataPath");
            return true;
        }

        World world = player.getWorld();
        Chunk chunk = world.getChunkAt(player.getLocation());

        try
        {
            player.sendMessage(ChatColor.YELLOW + "Collecting chunk data, please wait...");
            ChunkDataExtractor.writeChunkCsv(chunk, new File(setDataPath.getPath()));
            player.sendMessage(ChatColor.GREEN + "" + ChatColor.BOLD + "[!] Chunk data fetched successfully!");
        }
        catch (Exception e)
        {
            player.sendMessage(ChatColor.RED + "" + ChatColor.BOLD + "[!] Fatal error: " + e.getMessage());
            e.printStackTrace();
        }
        return true;
    }
}
