package com.ghosteau.generativeterrain.commands;

import ai.onnxruntime.*;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import org.bukkit.*;
import org.bukkit.block.Biome;
import org.bukkit.block.Block;
import org.bukkit.command.*;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;

public class modelGenerateTerrain implements CommandExecutor
{
    private final JavaPlugin plugin;
    private OrtEnvironment env;
    private OrtSession session;
    private final ConcurrentHashMap<UUID, AtomicBoolean> generationTasks = new ConcurrentHashMap<>();
    private final String model = "terrain_transformer_model.onnx";

    private static final int CHUNK_WIDTH = 16;
    private static final int CHUNK_DEPTH = 16;
    private static final int MODEL_CHUNK_HEIGHT = 32; // 32 for transformer, 256 for CNN model (as of now)
    private static final int MAX_Y = 319;
    private static final int MIN_Y = -64;
    private static final int WORLD_CHUNK_HEIGHT = MAX_Y - MIN_Y + 1;

    private static final int BLOCKS_PER_BATCH = 2048;
    private static final int TICKS_BETWEEN_BATCHES = 1;

    private final Map<String, Integer> biomeEncoder = new HashMap<>();
    private final Map<String, Integer> blockTypeEncoder = new HashMap<>();
    private final Map<Integer, Material> blockTypeDecoder = new HashMap<>();

    public modelGenerateTerrain(JavaPlugin plugin)
    {
        this.plugin = plugin;
        try
        {
            env = OrtEnvironment.getEnvironment();

            // Note: might be worth it to make a switch model command
            // There are two models -- transformer based and pure CNN based
            // Modify final variable "model" to use other model, default is transformer architecture
            File modelFile = new File(plugin.getDataFolder(), model);
            if (!modelFile.exists())
            {
                plugin.getLogger().severe("Model file not found! Please place name_model.onnx in the plugin's data folder.");
                return;
            }

            File mappingFile = new File(plugin.getDataFolder(), "block_id_mapping.json");
            blockTypeDecoder.putAll(loadBlockMapping(mappingFile));
            blockTypeEncoder.putAll(loadReverseBlockMapping(mappingFile));

            File biomeFile = new File(plugin.getDataFolder(), "biome_id_mapping.json");
            biomeEncoder.putAll(loadBiomeMapping(biomeFile));

            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setIntraOpNumThreads(2);
            sessionOptions.setInterOpNumThreads(2);
            sessionOptions.setMemoryPatternOptimization(true);
            session = env.createSession(modelFile.getAbsolutePath(), sessionOptions);
            plugin.getLogger().info("ONNX model loaded successfully!");
        }
        catch (Exception e)
        {
            plugin.getLogger().log(Level.SEVERE, "Error loading ONNX model", e);
        }
    }

    private Map<Integer, Material> loadBlockMapping(File jsonFile)
    {
        Map<Integer, Material> mapping = new HashMap<>();
        try (InputStream is = new FileInputStream(jsonFile))
        {
            String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
            for (String key : obj.keySet())
            {
                int index = Integer.parseInt(key);
                String blockName = obj.get(key).getAsString();
                Material mat = Material.matchMaterial(blockName);
                if (mat != null && mat.isBlock())
                {
                    mapping.put(index, mat);
                }
                else
                {
                    plugin.getLogger().warning("Invalid material in model mapping: " + blockName);
                }
            }
        }
        catch (IOException e)
        {
            plugin.getLogger().log(Level.SEVERE, "Failed to load block mapping", e);
        }

        plugin.getLogger().info("Loaded " + mapping.size() + " block mappings from JSON.");
        return mapping;
    }

    private Map<String, Integer> loadReverseBlockMapping(File jsonFile)
    {
        Map<String, Integer> reverseMapping = new HashMap<>();
        try (InputStream is = new FileInputStream(jsonFile))
        {
            String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
            for (String key : obj.keySet())
            {
                String blockName = obj.get(key).getAsString();
                reverseMapping.put(blockName, Integer.parseInt(key));
            }
        }
        catch (IOException e)
        {
            plugin.getLogger().log(Level.SEVERE, "Failed to load reverse block mapping", e);
        }
        return reverseMapping;
    }

    private Map<String, Integer> loadBiomeMapping(File jsonFile)
    {
        Map<String, Integer> mapping = new HashMap<>();
        try (InputStream is = new FileInputStream(jsonFile))
        {
            String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
            for (String key : obj.keySet())
            {
                mapping.put(key, obj.get(key).getAsInt());
            }
        }
        catch (IOException e)
        {
            plugin.getLogger().log(Level.SEVERE, "Failed to load biome mapping", e);
        }
        plugin.getLogger().info("Loaded " + mapping.size() + " biome mappings from JSON.");
        return mapping;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args)
    {
        if (!(sender instanceof Player))
        {
            sender.sendMessage(ChatColor.RED + "You must be in-game to execute this command.");
            return true;
        }

        Player player = (Player) sender;
        if (!player.hasPermission("generativeterrain.generateterrain"))
        {
            player.sendMessage(ChatColor.RED + "You don't have permission to use this command.");
            return true;
        }

        if (session == null)
        {
            player.sendMessage(ChatColor.RED + "The terrain generation model isn't loaded. Check server logs.");
            return true;
        }

        UUID playerUUID = player.getUniqueId();
        if (generationTasks.containsKey(playerUUID) && generationTasks.get(playerUUID).get())
        {
            player.sendMessage(ChatColor.RED + "You already have a terrain generation in progress. Type /generateterrain cancel to stop it.");
            return true;
        }

        // Parse arguments for chunk coordinates
        Chunk chunk = player.getLocation().getChunk();
        boolean fromArgs = false;

        if (args.length > 0 && args[0].equalsIgnoreCase("cancel"))
        {
            if (generationTasks.containsKey(playerUUID))
            {
                generationTasks.get(playerUUID).set(false);
                generationTasks.remove(playerUUID);
                player.sendMessage(ChatColor.YELLOW + "Terrain generation canceled.");
            }
            else
            {
                player.sendMessage(ChatColor.YELLOW + "You don't have any terrain generation in progress.");
            }

            return true;
        }

        if (args.length >= 2)
        {
            try
            {
                int chunkX = Integer.parseInt(args[0]);
                int chunkZ = Integer.parseInt(args[1]);
                chunk = player.getWorld().getChunkAt(chunkX, chunkZ);
                fromArgs = true;
            }
            catch (NumberFormatException e)
            {
                player.sendMessage(ChatColor.YELLOW + "Invalid chunk coordinates. Using current chunk.");
            }
        }

        generationTasks.put(playerUUID, new AtomicBoolean(true));

        player.sendMessage(ChatColor.GREEN + "Starting terrain generation for chunk: " +
                chunk.getX() + ", " + chunk.getZ() +
                (fromArgs ? "" : " (your current position)"));
        player.sendMessage(ChatColor.GRAY + "Type /generateterrain cancel to stop the generation.");

        // Start the async process
        startTerrainGeneration(chunk, player);
        return true;
    }

    private void startTerrainGeneration(Chunk chunk, Player player)
    {
        final UUID playerUUID = player.getUniqueId();
        final World world = chunk.getWorld();
        final int chunkX = chunk.getX() * 16;
        final int chunkZ = chunk.getZ() * 16;

        // Run data gathering and model inference in async task
        Bukkit.getScheduler().runTaskAsynchronously(plugin, () ->
        {
            try
            {
                if (!generationTasks.get(playerUUID).get()) return;

                player.sendMessage(ChatColor.AQUA + "Gathering data and preparing model input...");
                final String chunkBiomeName = getChunkBiome(chunk);

                // Create input tensor and fill with data from the world
                float[][][][][] inputTensorData = new float[1][10][CHUNK_WIDTH][MODEL_CHUNK_HEIGHT][CHUNK_DEPTH];
                fillInputTensor(inputTensorData, world, chunkX, chunkZ, chunkBiomeName);

                if (!generationTasks.get(playerUUID).get()) return;

                // Run inference
                player.sendMessage(ChatColor.AQUA + "Running AI model inference...");
                float[][][][] outputBlocks = runModelInference(inputTensorData, player);

                if (!generationTasks.get(playerUUID).get() || outputBlocks == null) return;

                // Start applying blocks in the main thread
                Bukkit.getScheduler().runTask(plugin, () ->
                        applyTerrainChanges(chunk, outputBlocks, player, playerUUID));

            }
            catch (Exception e)
            {
                generationTasks.get(playerUUID).set(false);
                player.sendMessage(ChatColor.RED + "" + ChatColor.BOLD + "[!] Error during terrain generation: " + e.getMessage());
                plugin.getLogger().log(Level.SEVERE, "Error in terrain generation", e);
            }
        });
    }

    private float[][][][] runModelInference(float[][][][][] inputTensorData, Player player)
    {
        try
        {
            // Convert tensor data to flat buffer for ONNX
            int totalSize = 1 * 10 * CHUNK_WIDTH * MODEL_CHUNK_HEIGHT * CHUNK_DEPTH;
            FloatBuffer inputBuffer = FloatBuffer.allocate(totalSize);

            // Flatten the tensor
            for (int c = 0; c < 10; c++)
                for (int x = 0; x < CHUNK_WIDTH; x++)
                    for (int y = 0; y < MODEL_CHUNK_HEIGHT; y++)
                        for (int z = 0; z < CHUNK_DEPTH; z++)
                            inputBuffer.put(inputTensorData[0][c][x][y][z]);
            inputBuffer.flip();

            // Create ONNX tensor
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer,
                    new long[]{1, 10, CHUNK_WIDTH, MODEL_CHUNK_HEIGHT, CHUNK_DEPTH});

            // Run inference
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input", inputTensor);

            OrtSession.Result result = null;
            try
            {
                result = session.run(inputs);
                OnnxTensor outputTensor = (OnnxTensor) result.get(0);

                // Process model output
                float[][][][][] outputRaw = (float[][][][][]) outputTensor.getValue();
                float[][][][] outputBlocks = new float[CHUNK_WIDTH][WORLD_CHUNK_HEIGHT][CHUNK_DEPTH][1];
                int numClasses = outputRaw[0].length;

                // Convert model height space to world height space
                for (int x = 0; x < CHUNK_WIDTH; x++)
                {
                    for (int modelY = 0; modelY < MODEL_CHUNK_HEIGHT; modelY++)
                    {
                        int worldY = modelY - (MODEL_CHUNK_HEIGHT - WORLD_CHUNK_HEIGHT) / 2;
                        if (worldY < 0 || worldY >= WORLD_CHUNK_HEIGHT) continue;

                        for (int z = 0; z < CHUNK_DEPTH; z++)
                        {
                            float maxProb = Float.NEGATIVE_INFINITY;
                            int bestClass = 0;

                            // Find highest probability class
                            for (int c = 0; c < numClasses; c++)
                            {
                                float prob = outputRaw[0][c][x][modelY][z];
                                if (prob > maxProb)
                                {
                                    maxProb = prob;
                                    bestClass = c;
                                }
                            }
                            outputBlocks[x][worldY][z][0] = bestClass;
                        }
                    }
                }

                return outputBlocks;
            }
            finally
            {
                // Clean up resources
                if (inputTensor != null) inputTensor.close();
                if (result != null) result.close();
            }
        }
        catch (OrtException e)
        {
            player.sendMessage(ChatColor.RED + "Model inference failed: " + e.getMessage());
            plugin.getLogger().log(Level.SEVERE, "Model inference error", e);
            return null;
        }
    }

    private void applyTerrainChanges(Chunk chunk, float[][][][] outputBlocks, Player player, UUID playerUUID)
    {
        final AtomicBoolean isGenerating = generationTasks.get(playerUUID);
        final int baseY = MIN_Y;
        final int totalBlocks = CHUNK_WIDTH * WORLD_CHUNK_HEIGHT * CHUNK_DEPTH;

        player.sendMessage(ChatColor.YELLOW + "Applying terrain changes to world...");

        // Use atomic values for thread safety
        final AtomicInteger blockIndex = new AtomicInteger(0);
        final AtomicInteger blocksChanged = new AtomicInteger(0);
        final AtomicInteger lastProgress = new AtomicInteger(-1);

        // Create a block update task that runs periodically
        new BukkitRunnable()
        {
            @Override
            public void run()
            {
                // Check if we should stop
                if (!isGenerating.get())
                {
                    player.sendMessage(ChatColor.YELLOW + "Terrain generation canceled.");
                    cancel();
                    return;
                }

                // Get the current block index
                int currentIndex = blockIndex.get();
                int processed = 0;

                // Process a batch of blocks
                while (currentIndex < totalBlocks && processed < BLOCKS_PER_BATCH)
                {
                    int x = (currentIndex / (WORLD_CHUNK_HEIGHT * CHUNK_DEPTH)) % CHUNK_WIDTH;
                    int y = (currentIndex / CHUNK_DEPTH) % WORLD_CHUNK_HEIGHT;
                    int z = currentIndex % CHUNK_DEPTH;

                    try
                    {
                        // Get block type from model output
                        int blockId = (int) outputBlocks[x][y][z][0];
                        Material mat = blockTypeDecoder.getOrDefault(blockId, Material.AIR);

                        // Get existing block
                        Block block = chunk.getBlock(x, y + baseY, z);
                        Material currentType = block.getType();

                        // Update block if different (skip air-to-air replacements)
                        if (mat != null && mat != currentType && !(mat == Material.AIR && currentType == Material.AIR))
                        {
                            // Safe block update with client notification
                            block.setType(mat, false);  // false = don't update physics for better performance
                            blocksChanged.incrementAndGet();
                        }
                    }
                    catch (Exception e)
                    {
                        plugin.getLogger().warning("Error setting block at " + x + "," + (y + baseY) + "," + z + ": " + e.getMessage());
                    }

                    processed++;
                    currentIndex = blockIndex.incrementAndGet();
                }

                // Report progress
                int progress = (int)((blockIndex.get() * 100.0) / totalBlocks);
                if (progress >= lastProgress.get() + 10 || blockIndex.get() >= totalBlocks)
                {
                    player.sendMessage(ChatColor.GRAY + "Progress: " + progress + "% (" +
                            blocksChanged.get() + " blocks changed)");
                    lastProgress.set(progress);
                }

                // Check if we're done
                if (blockIndex.get() >= totalBlocks)
                {
                    player.sendMessage(ChatColor.GREEN + "Terrain generation complete! Changed " +
                            blocksChanged.get() + " blocks.");
                    generationTasks.remove(playerUUID);
                    cancel();

                    // Notify client about block updates
                    chunk.getWorld().refreshChunk(chunk.getX(), chunk.getZ());
                }
            }
        }.runTaskTimer(plugin, 5L, TICKS_BETWEEN_BATCHES);
    }

    private void fillInputTensor(float[][][][][] input, World world, int chunkX, int chunkZ, String chunkBiomeName)
    {
        for (int x = 0; x < CHUNK_WIDTH; x++)
        {
            for (int modelY = 0; modelY < MODEL_CHUNK_HEIGHT; modelY++)
            {
                int offset = ((MODEL_CHUNK_HEIGHT - WORLD_CHUNK_HEIGHT) / 2);
                int worldY = modelY - offset + MIN_Y;

                for (int z = 0; z < CHUNK_DEPTH; z++)
                {
                    // Skip positions outside world bounds
                    if (worldY < MIN_Y || worldY > MAX_Y)
                    {
                        for (int c = 0; c < 10; c++)
                        {
                            input[0][c][x][modelY][z] = 0.0f;
                        }

                        continue;
                    }

                    // Get block and surrounding blocks
                    Block block = world.getBlockAt(chunkX + x, worldY, chunkZ + z);
                    Block leftBlock = (x > 0) ?
                            world.getBlockAt(chunkX + x - 1, worldY, chunkZ + z) :
                            world.getBlockAt(chunkX + x, worldY, chunkZ + z);
                    Block rightBlock = (x < CHUNK_WIDTH - 1) ?
                            world.getBlockAt(chunkX + x + 1, worldY, chunkZ + z) :
                            world.getBlockAt(chunkX + x, worldY, chunkZ + z);
                    Block belowBlock = (worldY > MIN_Y) ?
                            world.getBlockAt(chunkX + x, worldY - 1, chunkZ + z) : null;
                    Block aboveBlock = (worldY < MAX_Y) ?
                            world.getBlockAt(chunkX + x, worldY + 1, chunkZ + z) : null;
                    Block frontBlock = (z < CHUNK_DEPTH - 1) ?
                            world.getBlockAt(chunkX + x, worldY, chunkZ + z + 1) :
                            world.getBlockAt(chunkX + x, worldY, chunkZ + z);
                    Block behindBlock = (z > 0) ?
                            world.getBlockAt(chunkX + x, worldY, chunkZ + z - 1) :
                            world.getBlockAt(chunkX + x, worldY, chunkZ + z);

                    // Get biome name
                    String biomeName = block.getBiome().toString();

                    // Fill input tensor features
                    input[0][0][x][modelY][z] = getBiomeFeature(chunkBiomeName);
                    input[0][1][x][modelY][z] = getBiomeFeature(biomeName);
                    input[0][2][x][modelY][z] = (aboveBlock != null && aboveBlock.getType() == Material.AIR) ? 1.0f : 0.0f;
                    input[0][3][x][modelY][z] = block.getLightLevel() / 15.0f;
                    input[0][4][x][modelY][z] = leftBlock != null ? getBlockTypeFeature(leftBlock.getType().toString()) : 0.0f;
                    input[0][5][x][modelY][z] = rightBlock != null ? getBlockTypeFeature(rightBlock.getType().toString()) : 0.0f;
                    input[0][6][x][modelY][z] = belowBlock != null ? getBlockTypeFeature(belowBlock.getType().toString()) : 0.0f;
                    input[0][7][x][modelY][z] = aboveBlock != null ? getBlockTypeFeature(aboveBlock.getType().toString()) : 0.0f;
                    input[0][8][x][modelY][z] = frontBlock != null ? getBlockTypeFeature(frontBlock.getType().toString()) : 0.0f;
                    input[0][9][x][modelY][z] = behindBlock != null ? getBlockTypeFeature(behindBlock.getType().toString()) : 0.0f;
                }
            }
        }
    }

    private String getChunkBiome(Chunk chunk)
    {
        int centerX = chunk.getX() * 16 + 8;
        int centerZ = chunk.getZ() * 16 + 8;
        World world = chunk.getWorld();
        Biome biome = world.getBiome(centerX, 64, centerZ);
        return biome.toString();
    }

    private float getBiomeFeature(String biomeName)
    {
        return biomeEncoder.getOrDefault(biomeName, 0);
    }

    private float getBlockTypeFeature(String blockType)
    {
        return blockTypeEncoder.getOrDefault(blockType, 0);
    }

    public void cleanup()
    {
        try
        {
            // Cancel all running generation tasks
            for (Map.Entry<UUID, AtomicBoolean> entry : generationTasks.entrySet())
            {
                entry.getValue().set(false);
            }
            generationTasks.clear();

            // Close ONNX resources
            if (session != null)
            {
                session.close();
                session = null;
            }
            if (env != null)
            {
                env.close();
                env = null;
            }
            plugin.getLogger().info("ONNX resources cleaned up successfully");
        }
        catch (OrtException e)
        {
            plugin.getLogger().log(Level.WARNING, "Error closing ONNX resources", e);
        }
    }
}