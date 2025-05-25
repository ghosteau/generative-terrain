package com.ghosteau.generativeterrain.commands;

import org.bukkit.ChatColor;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

import java.io.File;

public class setDataPath implements CommandExecutor
{
    // Stores the user's input path.
    private static String userPath;

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args)
    {
        /*
        - The command to update the data extraction path, particularly used to extract data from Minecraft onto your PC in CSV format (refer to grabChunkData class for more).
        - Note that the path will always default to server if you simply just type in a file name; no spaces allowed at all in the path, including folder names.
        - As some of the warning messages imply, the directory is not based on your PC system root, rather the directory where the server is located.
        - It is important that the final part of the path (what your file will be named and saved as) ends with .csv to have data extracted correctly.
        */

        if (!(sender instanceof Player))
        {
            sender.sendMessage(ChatColor.RED + "You must be in-game to execute this command.");
            return true;
        }

        Player player = (Player)sender;
        if (!player.hasPermission("generativeterrain.setdatapath"))
        {
            player.sendMessage(ChatColor.RED + "You don't have permission to use this command.");
            return true;
        }

        if (cmd.getName().equalsIgnoreCase("setDataPath"))
        {
            if (args.length == 1)
            {
                String path = args[0];

                // Check if the path is valid
                if (!path.toLowerCase().endsWith(".csv"))
                {
                    player.sendMessage(ChatColor.RED + "Path must end with .csv extension");
                    return true;
                }

                // Attempt to validate the directory exists or can be created.
                File file = new File(path);
                File parentDir = file.getParentFile();

                // If path has a parent directory and it doesn't exist, check if we can create it.
                if (parentDir != null && !parentDir.exists())
                {
                    player.sendMessage(ChatColor.YELLOW + "Directory doesn't exist. It will be created when you run grabChunkData.");
                }

                userPath = path;
                player.sendMessage(ChatColor.GREEN + "" + ChatColor.BOLD + "[!] Path updated to: " + ChatColor.RESET + path);
            }
            else
            {
                player.sendMessage(ChatColor.RED + "Usage: /setDataPath <your/specified/path.csv>");
                player.sendMessage(ChatColor.YELLOW + "Ensure no spaces in path. Path is relative to server directory, not system root.");
            }
            return true;
        }
        return false;
    }

    public static String getPath()
    {
        // Returns the path string.
        return userPath;
    }
}