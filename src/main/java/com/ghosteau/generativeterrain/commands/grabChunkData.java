package com.ghosteau.generativeterrain.commands;

import org.bukkit.*;
import org.bukkit.block.BlockFace;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;

import java.io.*;
import java.util.ArrayList;

public class grabChunkData implements CommandExecutor
{
    // Height limit and minimum Y-value as of 1.21.4.
    final private int maxY = 319;
    final private int minY = -64;

    // Single biome label for the whole chunk.
    private String ChunkBiome = "UNKNOWN";

    // ArrayLists that store chunk data.
    private ArrayList<Integer> ListX = new ArrayList<>();
    private ArrayList<Integer> ListY = new ArrayList<>();
    private ArrayList<Integer> ListZ = new ArrayList<>();
    private ArrayList<String> Biome = new ArrayList<>();
    private ArrayList<String> Block_ID = new ArrayList<>();
    private ArrayList<Boolean> Is_Surface = new ArrayList<>();
    private ArrayList<Double> Light_Level = new ArrayList<>();
    private ArrayList<String> Block_to_Left = new ArrayList<>();
    private ArrayList<String> Block_to_Right = new ArrayList<>();
    private ArrayList<String> Block_Below = new ArrayList<>();
    private ArrayList<String> Block_Above = new ArrayList<>();
    private ArrayList<String> Block_in_Front = new ArrayList<>();
    private ArrayList<String> Block_Behind = new ArrayList<>();

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

        if (cmd.getName().equalsIgnoreCase("grabChunkData"))
        {
            // Check if path is set.
            if (setDataPath.getPath() == null || setDataPath.getPath().isEmpty())
            {
                player.sendMessage(ChatColor.RED + "Warning: Please set a path before using this command via /setDataPath");
                return true;
            }

            Location playerLocation = player.getLocation();
            World world = player.getWorld();
            Chunk playerChunk = world.getChunkAt(playerLocation);

            // Collect chunk data.
            try
            {
                player.sendMessage(ChatColor.YELLOW + "Collecting chunk data, please wait...");

                // Gets central biome for the chunk by finding the surface at the center block.
                int centerX = 8;
                int centerZ = 8;
                int surfaceY = maxY;

                while (surfaceY > minY && playerChunk.getBlock(centerX, surfaceY, centerZ).getType() == Material.AIR)
                {
                    surfaceY--;
                }

                ChunkBiome = playerChunk.getBlock(centerX, surfaceY, centerZ).getBiome().toString();

                // Loops through every block in the chunk.
                for (int x = 0; x < 16; x++)
                {
                    for (int y = minY; y <= maxY; y++)
                    {
                        for (int z = 0; z < 16; z++)
                        {
                            ListX.add(x);
                            ListY.add(y);
                            ListZ.add(z);
                            Biome.add(playerChunk.getBlock(x, y, z).getBiome().toString());
                            Block_ID.add(playerChunk.getBlock(x, y, z).getType().toString());
                            Is_Surface.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.UP).getType() == Material.AIR);
                            Light_Level.add((double)playerChunk.getBlock(x, y, z).getLightLevel());
                            Block_to_Left.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.WEST).getType().toString());
                            Block_to_Right.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.EAST).getType().toString());
                            Block_in_Front.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.NORTH).getType().toString());
                            Block_Behind.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.SOUTH).getType().toString());
                            Block_Below.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.DOWN).getType().toString());
                            Block_Above.add(playerChunk.getBlock(x, y, z).getRelative(BlockFace.UP).getType().toString());
                        }
                    }
                }

                // Export data to CSV.
                exportToCSV(setDataPath.getPath());
                clearAllLists();
                player.sendMessage(ChatColor.GREEN + "" + ChatColor.BOLD + "[!] Chunk data fetched successfully!");

                return true;
            }
            catch (Exception e)
            {
                player.sendMessage(ChatColor.RED + "" + ChatColor.BOLD + "[!] Fatal error: " + e.getMessage());
                e.printStackTrace();
                clearAllLists();
                return true;
            }
        }

        return false;
    }

    private void exportToCSV(String path)
    {
        // Exports all data from respective ArrayLists to CSV file format.
        final int length = ListY.size();
        BufferedWriter bufferedWriter = null;
        FileWriter fileWriter = null;

        try
        {
            File csvFile = new File(path);

            // Create parent directories if they don't exist.
            File parent = csvFile.getParentFile();
            if (parent != null && !parent.exists())
            {
                parent.mkdirs();
            }

            // Write data features and prepare to iterate
            fileWriter = new FileWriter(csvFile);
            bufferedWriter = new BufferedWriter(fileWriter);
            bufferedWriter.write("x,y,z,ChunkBiome,Biome,Block_ID,Is_Surface,Light_Level,Block_to_Left,Block_to_Right,Block_Below,Block_Above,Block_in_Front,Block_Behind");
            bufferedWriter.newLine();

            for (int i = 0; i < length; i++)
            {
                bufferedWriter.write(ListX.get(i).toString());
                bufferedWriter.write("," + ListY.get(i).toString());
                bufferedWriter.write("," + ListZ.get(i).toString());
                bufferedWriter.write("," + ChunkBiome);
                bufferedWriter.write("," + Biome.get(i));
                bufferedWriter.write("," + Block_ID.get(i));
                bufferedWriter.write("," + Is_Surface.get(i).toString());
                bufferedWriter.write("," + Light_Level.get(i).toString());
                bufferedWriter.write("," + Block_to_Left.get(i));
                bufferedWriter.write("," + Block_to_Right.get(i));
                bufferedWriter.write("," + Block_Below.get(i));
                bufferedWriter.write("," + Block_Above.get(i));
                bufferedWriter.write("," + Block_in_Front.get(i));
                bufferedWriter.write("," + Block_Behind.get(i));
                bufferedWriter.newLine();
            }
        }
        catch (IOException ioe)
        {
            System.err.println("Failed to export to CSV: " + ioe.getMessage());
            ioe.printStackTrace();
        }
        finally
        // Finally block always runs regardless of exception status; exists to close all file and buffered readers.
        {
            try
            {
                if (bufferedWriter != null)
                {
                    bufferedWriter.close();
                }

                if (fileWriter != null)
                {
                    fileWriter.close();
                }
            }
            catch (IOException ioe)
            {
                System.err.println("Error while closing BufferedWriter or FileWriter object(s).");
                ioe.printStackTrace();
            }
        }
    }

    private void clearAllLists()
    {
        // Clears all ArrayList data stored in the class.
        ListX.clear();
        ListY.clear();
        ListZ.clear();
        Block_ID.clear();
        Biome.clear();
        Is_Surface.clear();
        Light_Level.clear();
        Block_to_Left.clear();
        Block_to_Right.clear();
        Block_Above.clear();
        Block_Below.clear();
        Block_in_Front.clear();
        Block_Behind.clear();
        ChunkBiome = "UNKNOWN";
    }
}