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
import java.util.Random;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;

/**
 * /generateterrain -- generates a chunk with the trained conditional-VAE decoder.
 *
 * <h2>Why this changed from the transformer version</h2>
 * The old model conditioned each voxel on its neighbours' block IDs (plus light
 * and a surface flag). Those describe terrain that does not exist yet at
 * generation time, so the model only ever learned to copy existing neighbours
 * and could not generate from nothing. The replacement is a generative decoder
 * that takes only:
 * <ul>
 *   <li>a latent noise vector {@code z} (sampled fresh each run -> variety), and</li>
 *   <li>the chunk's biome id.</li>
 * </ul>
 * It outputs class logits over 27 block <i>groups</i> for every voxel; we argmax
 * to a group, then expand each group into a concrete block using local context
 * ({@link #expandGroupToBlock}).
 *
 * <h2>Files required in the plugin data folder</h2>
 * <ul>
 *   <li>{@code terrain_vae_decoder.onnx} -- the exported decoder</li>
 *   <li>{@code block_group_mapping.json} -- {@code {id: GROUP_NAME}}</li>
 *   <li>{@code biome_id_mapping.json}    -- {@code {BIOME_NAME: id}}</li>
 * </ul>
 * All three come from the training notebook's export step.
 */
public class modelGenerateTerrain implements CommandExecutor
{
    private final JavaPlugin plugin;
    private OrtEnvironment env;
    private OrtSession session;
    private final ConcurrentHashMap<UUID, AtomicBoolean> generationTasks = new ConcurrentHashMap<>();

    private final String model = "terrain_vae_decoder.onnx";

    private static final int CHUNK_WIDTH = 16;
    private static final int CHUNK_DEPTH = 16;
    private static final int MAX_Y = 319;
    private static final int MIN_Y = -64;
    private static final int WORLD_CHUNK_HEIGHT = MAX_Y - MIN_Y + 1; // 384

    // Latent input shape (with batch = 1), read from the ONNX graph so Java stays in
    // sync with the model. The baseline uses a vector latent [1, N]; the VAE uses a
    // small spatial latent [1, C, x, y, z]. We sample one N(0,1) value per element.
    private long[] zShape = {1, 64};

    private static final int BLOCKS_PER_BATCH = 2048;
    private static final int TICKS_BETWEEN_BATCHES = 1;

    // Mappings loaded from JSON.
    private final Map<String, Integer> biomeEncoder = new HashMap<>();   // biome name -> id
    private final Map<Integer, String> groupDecoder = new HashMap<>();   // class id -> group name

    private final Random random = new Random();

    public modelGenerateTerrain(JavaPlugin plugin)
    {
        this.plugin = plugin;
        try
        {
            env = OrtEnvironment.getEnvironment();

            File modelFile = new File(plugin.getDataFolder(), model);
            if (!modelFile.exists())
            {
                plugin.getLogger().severe("Model file not found! Please place " + model + " in the plugin's data folder.");
                return;
            }

            File groupFile = new File(plugin.getDataFolder(), "block_group_mapping.json");
            groupDecoder.putAll(loadGroupMapping(groupFile));

            File biomeFile = new File(plugin.getDataFolder(), "biome_id_mapping.json");
            biomeEncoder.putAll(loadBiomeMapping(biomeFile));

            OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
            sessionOptions.setIntraOpNumThreads(2);
            sessionOptions.setInterOpNumThreads(2);
            sessionOptions.setMemoryPatternOptimization(true);
            session = env.createSession(modelFile.getAbsolutePath(), sessionOptions);

            // Read the full "z" input shape from the graph and fix the batch dim to 1,
            // so we sample exactly the latent the loaded model expects (vector or spatial).
            NodeInfo zInfo = session.getInputInfo().get("z");
            if (zInfo != null && zInfo.getInfo() instanceof TensorInfo)
            {
                long[] shape = ((TensorInfo) zInfo.getInfo()).getShape();
                if (shape.length >= 1)
                {
                    zShape = shape.clone();
                    for (int i = 0; i < zShape.length; i++)
                        if (zShape[i] <= 0) zShape[i] = 1;   // dynamic dims (e.g. batch) -> 1
                }
            }

            plugin.getLogger().info("ONNX decoder loaded. zShape=" + java.util.Arrays.toString(zShape)
                    + ", groups=" + groupDecoder.size() + ", biomes=" + biomeEncoder.size());
        }
        catch (Exception e)
        {
            plugin.getLogger().log(Level.SEVERE, "Error loading ONNX model", e);
        }
    }

    /** Loads {@code {id: GROUP_NAME}} (the argmax class -> block group). */
    private Map<Integer, String> loadGroupMapping(File jsonFile)
    {
        Map<Integer, String> mapping = new HashMap<>();
        try (InputStream is = new FileInputStream(jsonFile))
        {
            String json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
            for (String key : obj.keySet())
            {
                mapping.put(Integer.parseInt(key), obj.get(key).getAsString());
            }
        }
        catch (IOException | NumberFormatException e)
        {
            plugin.getLogger().log(Level.SEVERE, "Failed to load block group mapping", e);
        }
        plugin.getLogger().info("Loaded " + mapping.size() + " block group mappings from JSON.");
        return mapping;
    }

    /** Loads {@code {BIOME_NAME: id}}. */
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

        // Run model inference in an async task; block placement happens back on the main thread.
        Bukkit.getScheduler().runTaskAsynchronously(plugin, () ->
        {
            try
            {
                if (!generationTasks.get(playerUUID).get()) return;

                final String chunkBiomeName = getChunkBiome(chunk);

                player.sendMessage(ChatColor.AQUA + "Running AI model inference...");
                int[][][] groupGrid = runModelInference(chunkBiomeName, player);

                if (!generationTasks.get(playerUUID).get() || groupGrid == null) return;

                // Start applying blocks in the main thread
                Bukkit.getScheduler().runTask(plugin, () ->
                        applyTerrainChanges(chunk, groupGrid, player, playerUUID));
            }
            catch (Exception e)
            {
                generationTasks.get(playerUUID).set(false);
                player.sendMessage(ChatColor.RED + "" + ChatColor.BOLD + "[!] Error during terrain generation: " + e.getMessage());
                plugin.getLogger().log(Level.SEVERE, "Error in terrain generation", e);
            }
        });
    }

    /**
     * Run the decoder once for this chunk.
     *
     * @return a [X][Y][Z] grid of block-group ids, or {@code null} on failure.
     */
    private int[][][] runModelInference(String chunkBiomeName, Player player)
    {
        // Inputs: a fresh latent (one N(0,1) value per element) and the chunk's biome id.
        long zCount = 1;
        for (long d : zShape) zCount *= d;
        FloatBuffer zBuffer = FloatBuffer.allocate((int) zCount);
        for (int i = 0; i < zCount; i++) zBuffer.put((float) random.nextGaussian());
        zBuffer.flip();
        long[] biomeData = new long[]{ biomeEncoder.getOrDefault(chunkBiomeName, 0) };

        OnnxTensor zTensor = null;
        OnnxTensor biomeTensor = null;
        OrtSession.Result result = null;
        try
        {
            zTensor = OnnxTensor.createTensor(env, zBuffer, zShape);  // FLOAT, model's z shape
            biomeTensor = OnnxTensor.createTensor(env, biomeData);   // INT64 [1]

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("z", zTensor);
            inputs.put("biome_id", biomeTensor);

            result = session.run(inputs);

            // Output: [1, numClasses, X, Y, Z]
            float[][][][][] outputRaw = (float[][][][][]) ((OnnxTensor) result.get(0)).getValue();
            int numClasses = outputRaw[0].length;

            int[][][] groupGrid = new int[CHUNK_WIDTH][WORLD_CHUNK_HEIGHT][CHUNK_DEPTH];
            for (int x = 0; x < CHUNK_WIDTH; x++)
            {
                for (int y = 0; y < WORLD_CHUNK_HEIGHT; y++)
                {
                    for (int z = 0; z < CHUNK_DEPTH; z++)
                    {
                        float maxProb = Float.NEGATIVE_INFINITY;
                        int bestClass = 0;
                        for (int c = 0; c < numClasses; c++)
                        {
                            float prob = outputRaw[0][c][x][y][z];
                            if (prob > maxProb)
                            {
                                maxProb = prob;
                                bestClass = c;
                            }
                        }
                        groupGrid[x][y][z] = bestClass;
                    }
                }
            }
            return groupGrid;
        }
        catch (OrtException e)
        {
            player.sendMessage(ChatColor.RED + "Model inference failed: " + e.getMessage());
            plugin.getLogger().log(Level.SEVERE, "Model inference error", e);
            return null;
        }
        finally
        {
            if (zTensor != null) zTensor.close();
            if (biomeTensor != null) biomeTensor.close();
            if (result != null) result.close();
        }
    }

    private void applyTerrainChanges(Chunk chunk, int[][][] groupGrid, Player player, UUID playerUUID)
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
                        // Decode group id -> group name -> concrete block (height/context aware)
                        int groupId = groupGrid[x][y][z];
                        String groupName = groupDecoder.getOrDefault(groupId, "AIR");
                        Material mat = expandGroupToBlock(groupName, chunk, x, y + baseY, z);

                        Block block = chunk.getBlock(x, y + baseY, z);
                        Material currentType = block.getType();

                        // Update block if different (skip air-to-air replacements)
                        if (mat != null && mat != currentType && !(mat == Material.AIR && currentType == Material.AIR))
                        {
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

                    chunk.getWorld().refreshChunk(chunk.getX(), chunk.getZ());
                }
            }
        }.runTaskTimer(plugin, 5L, TICKS_BETWEEN_BATCHES);
    }

    /**
     * Expand a predicted block group into a concrete {@link Material}, using local
     * context the model does not predict (deepslate vs stone ores by height,
     * logs vs leaves by neighbours).
     */
    private Material expandGroupToBlock(String groupName, Chunk chunk, int x, int y, int z)
    {
        World world = chunk.getWorld();
        Block above = (y < MAX_Y) ? world.getBlockAt(x, y + 1, z) : null;
        Block below = (y > MIN_Y) ? world.getBlockAt(x, y - 1, z) : null;

        switch (groupName)
        {
            case "AIR":
            case "CAVE_AIR":
                return Material.AIR;

            case "STONE":      return Material.STONE;
            case "DEEPSLATE":  return Material.DEEPSLATE;
            case "DIRT":       return Material.DIRT;
            case "GRASS":      return Material.GRASS_BLOCK;
            case "SAND":       return Material.SAND;
            case "GRAVEL":     return Material.GRAVEL;
            case "CLAY":       return Material.CLAY;
            case "WATER":      return Material.WATER;
            case "LAVA":       return Material.LAVA;
            case "BEDROCK":    return Material.BEDROCK;

            case "COAL_ORE":     return (y < 0) ? Material.DEEPSLATE_COAL_ORE : Material.COAL_ORE;
            case "IRON_ORE":     return (y < 0) ? Material.DEEPSLATE_IRON_ORE : Material.IRON_ORE;
            case "COPPER_ORE":   return (y < 0) ? Material.DEEPSLATE_COPPER_ORE : Material.COPPER_ORE;
            case "GOLD_ORE":     return (y < 0) ? Material.DEEPSLATE_GOLD_ORE : Material.GOLD_ORE;
            case "REDSTONE_ORE": return (y < 0) ? Material.DEEPSLATE_REDSTONE_ORE : Material.REDSTONE_ORE;
            case "LAPIS_ORE":    return (y < 0) ? Material.DEEPSLATE_LAPIS_ORE : Material.LAPIS_ORE;
            case "DIAMOND_ORE":  return (y < 0) ? Material.DEEPSLATE_DIAMOND_ORE : Material.DIAMOND_ORE;
            case "EMERALD_ORE":  return (y < 0) ? Material.DEEPSLATE_EMERALD_ORE : Material.EMERALD_ORE;

            case "OAK_WOOD":
                return hasWoodNearby(chunk, x, y, z) && hasAirAbove(above) ? Material.OAK_LEAVES : Material.OAK_LOG;
            case "SPRUCE_WOOD":
                return hasWoodNearby(chunk, x, y, z) && hasAirAbove(above) ? Material.SPRUCE_LEAVES : Material.SPRUCE_LOG;
            case "BIRCH_WOOD":
                return hasWoodNearby(chunk, x, y, z) && hasAirAbove(above) ? Material.BIRCH_LEAVES : Material.BIRCH_LOG;
            case "JUNGLE_WOOD":
                return hasWoodNearby(chunk, x, y, z) && hasAirAbove(above) ? Material.JUNGLE_LEAVES : Material.JUNGLE_LOG;
            case "ACACIA_WOOD":
                return hasWoodNearby(chunk, x, y, z) && hasAirAbove(above) ? Material.ACACIA_LEAVES : Material.ACACIA_LOG;
            case "DARK_OAK_WOOD":
                return hasWoodNearby(chunk, x, y, z) && hasAirAbove(above) ? Material.DARK_OAK_LEAVES : Material.DARK_OAK_LOG;

            case "VEGETATION":
                return isSolid(below) ? Material.SHORT_GRASS : Material.AIR;

            case "MISC":
            default:
                return Material.STONE;
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

    private boolean hasWoodNearby(Chunk chunk, int x, int y, int z)
    {
        int range = 2;
        for (int dx = -range; dx <= range; dx++)
        {
            for (int dy = -range; dy <= range; dy++)
            {
                for (int dz = -range; dz <= range; dz++)
                {
                    try
                    {
                        String name = chunk.getWorld().getBlockAt(x + dx, y + dy, z + dz).getType().toString();
                        if (name.contains("LOG")) return true;
                    }
                    catch (Exception ignored) { }
                }
            }
        }
        return false;
    }

    private boolean hasAirAbove(Block block)
    {
        return block != null && block.getType() == Material.AIR;
    }

    private boolean isSolid(Block block)
    {
        return block != null && block.getType().isSolid();
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
