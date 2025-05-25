package com.ghosteau.generativeterrain;

import com.ghosteau.generativeterrain.commands.grabChunkData;
import com.ghosteau.generativeterrain.commands.modelGenerateTerrain;
import com.ghosteau.generativeterrain.commands.setDataPath;

import org.bukkit.ChatColor;
import org.bukkit.plugin.java.JavaPlugin;

public class GenerativeTerrain extends JavaPlugin
{
    private modelGenerateTerrain terrainGenerator;

    @Override
    public void onEnable()
    {
        // Register commands
        this.getCommand("grabChunkData").setExecutor(new grabChunkData());
        this.getCommand("setDataPath").setExecutor(new setDataPath());

        // Initialize and register the terrain generator command
        terrainGenerator = new modelGenerateTerrain(this);
        this.getCommand("generateTerrain").setExecutor(terrainGenerator);

        // Create data folder if it doesn't exist
        if (!getDataFolder().exists())
        {
            getDataFolder().mkdirs();
        }

        // Just a message to let server know plugin is enabled
        getServer().getConsoleSender().sendMessage(ChatColor.DARK_AQUA + "[GenerativeTerrain]: Plugin enabled.");
        getServer().getConsoleSender().sendMessage(ChatColor.DARK_AQUA + "[GenerativeTerrain]: Loading ML model...");
    }

    @Override
    public void onDisable()
    {
        // Clean up ONNX resources
        if (terrainGenerator != null)
        {
            terrainGenerator.cleanup();
        }

        // Just a message to let server know plugin is disabled
        getServer().getConsoleSender().sendMessage(ChatColor.DARK_AQUA + "[GenerativeTerrain]: Plugin disabled.");
    }
}
