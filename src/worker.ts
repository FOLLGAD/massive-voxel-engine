/// <reference types="bun-types" />

import { vec3 } from "gl-matrix";
import { Terrain } from "./terrain";
import { Chunk } from "./chunk";
import { ENABLE_GREEDY_MESHING } from "./config";
import { ChunkStorage } from "./chunk-storage";
import { getChunkKey } from "./chunk";
import log from "./logger";

// Initialize terrain generator and storage
const terrain = new Terrain();
const chunkStorage = new ChunkStorage();

// Initialize storage
let storageAvailable = false;
let storageInitialized = false;
const storageInitPromise = chunkStorage.ensureInitialized().then(async () => {
  storageAvailable = true;
  storageInitialized = true;
  console.log("Worker", "✅ Worker storage initialized successfully - IndexedDB is available");
}).catch(error => {
  storageAvailable = false;
  storageInitialized = true; // Still mark as "done" even if failed
  console.error("Worker", "❌ Failed to initialize worker storage - running without persistence:", error);
});

// Helper function to wait for storage initialization
const waitForStorageInit = async () => {
  if (!storageInitialized) {
    log("Worker", "⏳ Waiting for storage initialization...");
    await storageInitPromise;
  }
};

// Batched storage system for modified chunks
const pendingStorageSaves = new Map<string, { position: vec3, data: Uint8Array, timestamp: number }>();
const STORAGE_BATCH_INTERVAL = 2000; // Save every 2 seconds
const MAX_PENDING_SAVES = 100; // Force flush if too many pending

// Periodic flush of pending saves
const flushPendingSaves = async () => {
  if (pendingStorageSaves.size === 0) {
    log("Worker", "📦 No pending saves to flush");
    return;
  }
  
  const chunksToSave = Array.from(pendingStorageSaves.values());
  pendingStorageSaves.clear();
  
  log("Worker", `💾 Attempting to flush ${chunksToSave.length} chunks to storage. Storage available: ${storageAvailable}`);
  
  if (storageAvailable) {
    try {
      // Save all pending chunks
      let savedCount = 0;
      for (const { position, data } of chunksToSave) {
        await chunkStorage.saveChunk(position, data);
        savedCount++;
      }
      log("Worker", `✅ Successfully saved ${savedCount}/${chunksToSave.length} chunks to IndexedDB storage`);
    } catch (error) {
      log.error("Worker", "❌ Failed to batch save chunks to storage:", error);
    }
  } else {
    log.warn("Worker", `⚠️ Storage not available - discarding ${chunksToSave.length} pending saves`);
  }
};

// Start periodic flush
setInterval(() => {
  log("Worker", `⏰ Periodic flush check (every ${STORAGE_BATCH_INTERVAL}ms)`);
  flushPendingSaves();
}, STORAGE_BATCH_INTERVAL);

self.onmessage = async (event: MessageEvent) => {
  const type = event.data.type;
  const _position = event.data.position;
  
  // Some message types don't require position (like unloadChunks, getStorageStats, flushPendingSaves, getStorageStatus)
  if (!_position && type !== "unloadChunks" && type !== "getStorageStats" && type !== "flushPendingSaves" && type !== "getStorageStatus") {
    log.warn("Worker", "No position provided in message");
    return;
  }

  const position = _position ? vec3.fromValues(_position[0], _position[1], _position[2]) : null;
  
  if (type === "requestChunk") {
    if (!position) {
      log.error("Worker", "requestChunk requires position");
      return;
    }
    
    try {
      // Wait for storage initialization before proceeding
      await waitForStorageInit();
      
      // First try to load from storage if available
      let chunk: Chunk;
      let existingData: Uint8Array | null = null;
      
      if (storageAvailable) {
        try {
          existingData = await chunkStorage.loadChunk(position);
        } catch (error) {
          log.warn("Worker", `Storage load failed for ${position[0]},${position[1]},${position[2]}, generating new:`, error);
        }
      }
      
      if (existingData) {
        chunk = new Chunk(position, existingData);
        log("Worker", `📦 Loaded chunk from storage for position ${position[0]},${position[1]},${position[2]}`);
      } else {
        log("Worker", `🎲 Generating new terrain for position ${position[0]},${position[1]},${position[2]} (not in storage)`);
        
        // Generate new terrain
        chunk = terrain.generateTerrain(position);
        
        // Save to storage if available
        if (storageAvailable) {
          try {
            await chunkStorage.saveChunk(position, chunk.data);
            log("Worker", `Saved chunk to storage for position ${position[0]},${position[1]},${position[2]}`);
          } catch (error) {
            log.warn("Worker", `Storage save failed for ${position[0]},${position[1]},${position[2]}:`, error);
          }
        } else {
          log("Worker", `Generated chunk for position ${position[0]},${position[1]},${position[2]} (no storage)`);
        }
      }

      log("Worker", `Generating mesh for position ${position[0]},${position[1]},${position[2]}`);
      
      const mesh = chunk.generateMesh();
      log("Worker", `Mesh generated: ${mesh.vertices.length} vertices, ${mesh.indices.length} indices`);

      // Send chunk data first for physics
      self.postMessage({
        type: "chunkDataAvailable",
        position,
        voxels: chunk.data,
      });

      // Then send mesh data for rendering
      self.postMessage({
        type: "chunkMeshUpdated",
        position,
        vertices: mesh.vertices.buffer,
        indices: mesh.indices.buffer,
        visibilityBits: chunk.generateVisibilityMatrix(),
      });
      
    } catch (error) {
      console.error("❌ Worker error during processing:", error);
      log.error("Worker", "Error during chunk processing:", error);
    }
    
  } else if (type === "requestChunkData") {
    if (!position) {
      log.error("Worker", "requestChunkData requires position");
      return;
    }
    
    // Handle requests for chunk data (for physics, etc.)
    try {
      // Wait for storage initialization before proceeding
      await waitForStorageInit();
      
      let chunkData: Uint8Array | null = null;
      
      if (storageAvailable) {
        try {
          chunkData = await chunkStorage.loadChunk(position);
        } catch (error) {
          log.warn("Worker", `Storage load failed for chunk data ${position[0]},${position[1]},${position[2]}:`, error);
        }
      }
      
      if (chunkData) {
        self.postMessage({
          type: "chunkDataAvailable",
          position,
          voxels: chunkData,
        });
        log("Worker", `📦 Sent stored chunk data for physics request ${position[0]},${position[1]},${position[2]}`);
      } else {
        // If no data in storage, generate it
        log("Worker", `🎲 Generating chunk data for physics request ${position[0]},${position[1]},${position[2]} (not in storage)`);
        const chunk = terrain.generateTerrain(position);
        
        // Save to storage if available
        if (storageAvailable) {
          try {
            await chunkStorage.saveChunk(position, chunk.data);
          } catch (error) {
            log.warn("Worker", `Storage save failed after generation for ${position[0]},${position[1]},${position[2]}:`, error);
          }
        }
        
        self.postMessage({
          type: "chunkDataAvailable",
          position,
          voxels: chunk.data,
        });
      }
    } catch (error) {
      console.error("❌ Worker error loading chunk data:", error);
      log.error("Worker", "Error loading chunk data:", error);
      self.postMessage({
        type: "chunkDataNotFound",
        position,
      });
    }
    
  } else if (type === "renderChunk") {
    // Handle modified chunk re-rendering
    try {
      const { position, data: dataBuffer, immediate = false } = event.data;
      const data = new Uint8Array(dataBuffer);
      
      // Queue chunk for batched storage save
      const key = getChunkKey(position);
      pendingStorageSaves.set(key, {
        position: position,
        data: data.slice(), // Make a copy to avoid issues with buffer reuse
        timestamp: Date.now()
      });
      
      log("Worker", `📝 Queued chunk ${key} for storage (${pendingStorageSaves.size} pending). Immediate: ${immediate}`);
      
      // For user modifications (block place/destroy), flush immediately
      if (immediate) {
        log("Worker", `🚀 Immediate flush for user modification of chunk ${key}`);
        await flushPendingSaves();
      }
      // Otherwise, force flush if we have too many pending saves
      else if (pendingStorageSaves.size >= MAX_PENDING_SAVES) {
        log("Worker", `🔄 Force flushing ${pendingStorageSaves.size} pending saves (max limit reached)`);
        await flushPendingSaves();
      }
      
      const chunk = new Chunk(position, data);
      const mesh = chunk.generateMesh();

      self.postMessage({
        type: "chunkMeshUpdated",
        position: position,
        vertices: mesh.vertices.buffer,
        indices: mesh.indices.buffer,
        visibilityBits: chunk.generateVisibilityMatrix(),
      });
    } catch (error) {
      console.error("❌ Worker error during chunk re-rendering:", error);
      log.error("Worker", "Error during chunk re-rendering:", error);
    }
    
  } else if (type === "deleteChunk") {
    if (!position) {
      log.error("Worker", "deleteChunk requires position");
      return;
    }
    
    // Handle explicit chunk deletion from storage
    // NOTE: This is for explicit deletions only, not automatic unloading
    // Automatic unloading never deletes from storage
    try {
      await chunkStorage.deleteChunk(position);
      self.postMessage({
        type: "chunkDeleted",
        position,
      });
    } catch (error) {
      console.error("❌ Worker error deleting chunk:", error);
      log.error("Worker", "Error deleting chunk:", error);
    }
    
  } else if (type === "getStorageStats") {
    // Handle storage stats requests
    try {
      const stats = await chunkStorage.getStats();
      self.postMessage({
        type: "storageStats",
        stats,
      });
    } catch (error) {
      console.error("❌ Worker error getting storage stats:", error);
      log.error("Worker", "Error getting storage stats:", error);
    }
    
  } else if (type === "unloadChunks") {
    // Handle chunk unloading requests
    // NOTE: This only identifies chunks for mesh unloading, NOT storage deletion
    // Storage is permanent and never deleted during unloading
    try {
      const { allChunkKeys, playerPosition, loadRadiusXZ, loadRadiusY, unloadBufferXZ, unloadBufferY } = event.data;
      const chunksToUnload: string[] = [];
      
      // Calculate expanded unload radius (load radius + buffer)
      const unloadRadiusXZ = loadRadiusXZ + unloadBufferXZ;
      const unloadRadiusY = loadRadiusY + unloadBufferY;
      
      // Check each chunk to see if it should be unloaded (mesh only)
      for (const chunkKey of allChunkKeys) {
        const [x, y, z] = chunkKey.split(',').map(Number);
        const chunkPos = vec3.fromValues(x, y, z);
        
        // Calculate distance from player
        const dx = Math.abs(chunkPos[0] - playerPosition[0]);
        const dy = Math.abs(chunkPos[1] - playerPosition[1]);
        const dz = Math.abs(chunkPos[2] - playerPosition[2]);
        
        // Check if chunk is outside unload radius
        if (dx > unloadRadiusXZ || dy > unloadRadiusY || dz > unloadRadiusXZ) {
          chunksToUnload.push(chunkKey);
        }
      }
      
      // Send list of chunks to unload back to main thread (mesh unloading only)
      if (chunksToUnload.length > 0) {
        self.postMessage({
          type: "chunksToUnload",
          chunks: chunksToUnload,
        });
        log("Worker", `Identified ${chunksToUnload.length} chunks for mesh unloading`);
      }
    } catch (error) {
      console.error("❌ Worker error during chunk unloading:", error);
      log.error("Worker", "Error during chunk unloading:", error);
    }
    
  } else if (type === "flushPendingSaves") {
    // Handle manual flush requests
    try {
      await flushPendingSaves();
      self.postMessage({
        type: "flushComplete",
      });
    } catch (error) {
      console.error("❌ Worker error flushing pending saves:", error);
      log.error("Worker", "Error flushing pending saves:", error);
    }
    
  } else if (type === "getStorageStatus") {
    // Handle storage status requests for debugging
    try {
      self.postMessage({
        type: "storageStatus",
        status: {
          initialized: storageInitialized,
          available: storageAvailable,
          pendingSaves: pendingStorageSaves.size,
          interval: STORAGE_BATCH_INTERVAL,
          maxPending: MAX_PENDING_SAVES
        }
      });
    } catch (error) {
      console.error("❌ Worker error getting storage status:", error);
      log.error("Worker", "Error getting storage status:", error);
    }
    
  } else {
    log.warn("Worker", `Unknown message type received: ${type}`);
  }
};

// Note: Web workers don't support beforeunload events
// The main thread must handle cleanup before worker termination

log(
  "Worker",
  `Worker script initialized with storage. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
