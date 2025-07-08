/// <reference types="bun-types" />
/// <reference types="@webgpu/types" />

import { vec3 } from "gl-matrix"; // Keep gl-matrix for look direction vector
import { ENABLE_GREEDY_MESHING, LOAD_RADIUS_XZ, LOAD_RADIUS_Y } from "./config";
import {
  CHUNK_SIZE_X,
  CHUNK_SIZE_Y,
  CHUNK_SIZE_Z,
  UNLOAD_BUFFER_XZ,
  UNLOAD_BUFFER_Y,
  PHYSICS_CACHE_RADIUS_XZ,
  PHYSICS_CACHE_RADIUS_Y,
} from "./config";
import { PlayerState, updatePhysics } from "./physics"; // Import physics
import { Renderer } from "./renderer"; // Import renderer
import {
  Chunk,
  getChunkKey,
  getChunkOfPosition,
  getLocalPosition,
} from "./chunk";
import log from "./logger";
import { KeyboardState } from "./keyboard";
import { VoxelType } from "./common/voxel-types";
import { WorkerManager } from "./worker-manager";
import { createAABB } from "./aabb";
let debugMode = false;
log("Main", "Main script loaded.");

const FACE_NORMALS = {
  [0]: vec3.fromValues(1, 0, 0),
  [1]: vec3.fromValues(-1, 0, 0),
  [2]: vec3.fromValues(0, 1, 0),
  [3]: vec3.fromValues(0, -1, 0),
  [4]: vec3.fromValues(0, 0, 1),
  [5]: vec3.fromValues(0, 0, -1),
};

// --- Global State ---
const requestedChunkKeys = new Set<string>();
const chunkDataCache = new Map<string, Uint8Array>(); // Simple cache for chunk data needed by physics
const playerState = new PlayerState();
let rendererState: Renderer; // Will be initialized later
let chunksReceived = 0; // Counter for received chunks

// --- Camera/Input State ---
let cameraYaw = Math.PI / 4;
let cameraPitch = -Math.PI / 8;
const MOUSE_SENSITIVITY = 0.005;
const keyboardState = new KeyboardState();

// --- FPS Calculation State ---
const frameTimes: number[] = [];
const maxFrameSamples = 60;

async function main() {
  const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
  if (!canvas) {
    log.error("Main", "Canvas element not found!");
    return;
  }
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  // Prevent default context menu on right-click (optional)
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  canvas.addEventListener("click", async () => {
    await canvas.requestPointerLock({
      // unadjustedMovement: true,
    });
  });

  const updatePosition = (e: MouseEvent) => {
    const deltaX = e.movementX;
    const deltaY = e.movementY;
    cameraYaw -= deltaX * MOUSE_SENSITIVITY;
    cameraPitch -= deltaY * MOUSE_SENSITIVITY;
    const pitchLimit = Math.PI / 2 - 0.01;
    cameraPitch = Math.max(-pitchLimit, Math.min(pitchLimit, cameraPitch));
  };

  // capture mouse movement
  document.addEventListener("pointerlockchange", () => {
    if (document.pointerLockElement === canvas) {
      document.addEventListener("mousemove", updatePosition, false);
    } else {
      document.removeEventListener("mousemove", updatePosition, false);
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    keyboardState.pressedKeys.add(e.code);
    keyboardState.downKeys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    keyboardState.downKeys.delete(e.code);
  });
  window.addEventListener("mousedown", (e) => {
    const isRightClick = e.button === 2;
    if (isRightClick) {
      keyboardState.mouseRightClicked = true;
    } else {
      keyboardState.mouseDown = true;
      keyboardState.mouseClicked = true;
    }
  });
  window.addEventListener("mouseup", () => {
    keyboardState.mouseDown = false;
  });

  // --- Initialize Renderer ---
  try {
    rendererState = await Renderer.create(canvas);
    log("Main", "Renderer Initialized");
  } catch (error) {
    log("Main", "Failed to initialize renderer:", error);
    alert(
      "Failed to initialize renderer. You need WebGPU enabled in your browser."
    );
    return; // Stop if renderer fails
  }

  // Initialize worker pool
  const numWorkers = navigator.hardwareConcurrency || 4;
  log("Main", `Initializing ${numWorkers} workers...`);

  const workerManager = new WorkerManager(numWorkers);

  // Workers now handle their own storage, no need for hybrid chunk manager
  log("Main", "Workers will handle their own storage");

  let blockToPlace: VoxelType = VoxelType.STONE;

    // Function to request chunk data from workers when needed for physics
  const requestChunkData = (position: vec3): Promise<Uint8Array | null> => {
    return new Promise((resolve, reject) => {
      const key = getChunkKey(position);
      
      // Check cache first
      if (chunkDataCache.has(key)) {
        resolve(chunkDataCache.get(key)!);
        return;
      }
      
      // Set up one-time listener for the response
      const responseHandler = (event: MessageEvent) => {
        if (event.data.type === "chunkDataAvailable" && getChunkKey(event.data.position) === key) {
          const voxelData = event.data.voxels as Uint8Array;
          chunkDataCache.set(key, voxelData);
          resolve(voxelData);
          workerManager.removeMessageHandler(responseHandler);
        } else if (event.data.type === "chunkDataNotFound" && getChunkKey(event.data.position) === key) {
          resolve(null);
          workerManager.removeMessageHandler(responseHandler);
        }
      };
      
      workerManager.addMessageHandler(responseHandler);
      
      // Request chunk data from worker
      workerManager.queueTask({
        type: "requestChunkData",
        position,
      });
      
      // Add timeout
      setTimeout(() => {
        workerManager.removeMessageHandler(responseHandler);
        reject(new Error(`Timeout requesting chunk data for ${key}`));
      }, 5000);
    });
  };

  // --- Worker Message Handling ---
  const workerMessageHandler = (event: MessageEvent) => {
    const type = event.data.type;
    log("Worker", `Received message type: ${type}`);

    if (type === "chunkDataAvailable") {
      // Handle chunk data responses for physics (cache them)
      const { position, voxels } = event.data;
      const key = getChunkKey(position);
      const voxelData = voxels as Uint8Array;
      chunkDataCache.set(key, voxelData);
      chunksReceived++;
      
      // Mark chunk as requested (it's now available)
      requestedChunkKeys.add(key);
      log("Debug", `Cached chunk data for ${key}, total received: ${chunksReceived}`);
    } else if (type === "chunkMeshUpdated") {
      if (!rendererState) throw new Error("Renderer not found");

              const {
          position,
          vertices: verticesBuffer,
          indices: indicesBuffer,
          visibilityBits,
        } = event.data as {
          position: vec3;
          vertices: Float32Array;
          indices: Uint32Array;
          visibilityBits: number;
        };
        const vertices = new Float32Array(verticesBuffer);
        const indices = new Uint32Array(indicesBuffer);

        // Workers now handle their own storage, no need to save chunk data here

      const minX = position[0] * CHUNK_SIZE_X;
      const minY = position[1] * CHUNK_SIZE_Y;
      const minZ = position[2] * CHUNK_SIZE_Z;
      const maxX = minX + CHUNK_SIZE_X;
      const maxY = minY + CHUNK_SIZE_Y;
      const maxZ = minZ + CHUNK_SIZE_Z;
      const aabb = createAABB(
        vec3.fromValues(minX, minY, minZ),
        vec3.fromValues(maxX, maxY, maxZ)
      );

      try {
        rendererState.chunkManager.updateChunkGeometryInfo(
          position,
          vertices,
          vertices.byteLength,
          indices,
          indices.byteLength,
          aabb,
          visibilityBits
        );
        log("Debug", `Updated chunk geometry for ${getChunkKey(position)}, vertices: ${vertices.length}, indices: ${indices.length}`);
      } catch (error) {
        const key = getChunkKey(position);
        log.error("Main", `Error processing mesh update for ${key}:`, error);
      }
    } else if (type === "chunksToUnload") {
      const { chunks } = event.data;
      
      if (chunks.length > 0) {
        // Add chunks to unload queue for batched processing
        chunksToUnloadQueue.push(...chunks);
        log("Main", `Queued ${chunks.length} chunks for unloading (${chunksToUnloadQueue.length} total in queue)`);
      }
    } else if (type === "chunkDataNotFound") {
      const { position } = event.data;
      const key = getChunkKey(position);
      log("Main", `Chunk data not found for ${key}`);
      // This is handled by the requestChunkData promise rejection
    } else if (type === "storageStats") {
      const { stats } = event.data;
      log("Main", "Worker Storage Stats:", stats);
    } else if (type === "flushComplete") {
      log("Main", "Worker completed flushing pending saves to storage");
    } else if (type === "storageStatus") {
      const { status } = event.data;
      log("Main", "Worker Storage Status:", status);
    } else {
      log.warn("Main", `Unknown message type from worker: ${type}`);
    }
  };
  workerManager.setMessageHandler(workerMessageHandler);

  // Debug Info Element
  const debugInfoElement = document.getElementById(
    "debug-info"
  ) as HTMLDivElement;

  const toolbarElement = document.getElementById("toolbar") as HTMLDivElement;

  const getBlockLookedAt = (position: vec3, cameraYaw: number) => {
    const lookDirection = vec3.fromValues(
      Math.cos(cameraPitch) * Math.sin(cameraYaw),
      Math.sin(cameraPitch),
      Math.cos(cameraPitch) * Math.cos(cameraYaw)
    );
    vec3.normalize(lookDirection, lookDirection);

    const rayStart = vec3.clone(position);

    // Raycast to find the block the player is looking at
    const MAX_DISTANCE = 20.0; // Maximum distance to check for blocks
    const STEP_SIZE = 0.01; // Size of each step along the ray

    const currentPos = vec3.clone(rayStart);
    const lastPos = vec3.clone(currentPos);

    // Step along the ray
    for (let distance = 0; distance <= MAX_DISTANCE; distance += STEP_SIZE) {
      vec3.scaleAndAdd(currentPos, rayStart, lookDirection, distance);

      // Get block at current position
      const block = vec3.fromValues(
        Math.floor(currentPos[0]),
        Math.floor(currentPos[1]),
        Math.floor(currentPos[2])
      );

      // Get chunk key for this block
      const blockChunk = getChunkOfPosition(block);
      const chunkKey = getChunkKey(blockChunk);

      // Check if chunk is loaded (only from cache for performance)
      const key = getChunkKey(blockChunk);
      const cached = chunkDataCache.get(key);
      if (!cached) continue;

      // Get local coordinates within chunk
      const localPosition = getLocalPosition(block);

      // Get voxel index
      const chunk = new Chunk(blockChunk, cached);

      // Check if block exists (non-zero)
      if (chunk.getVoxel(localPosition) !== VoxelType.AIR) {
        // --- Determine which face was hit using Ray-AABB intersection ---
        const aabbMin = block;
        const aabbMax = vec3.add(vec3.create(), block, [1, 1, 1]);
        // Avoid division by zero if lookDirection component is zero
        const invDirX =
          lookDirection[0] === 0
            ? Number.POSITIVE_INFINITY
            : 1 / lookDirection[0];
        const invDirY =
          lookDirection[1] === 0
            ? Number.POSITIVE_INFINITY
            : 1 / lookDirection[1];
        const invDirZ =
          lookDirection[2] === 0
            ? Number.POSITIVE_INFINITY
            : 1 / lookDirection[2];

        const t1x = (aabbMin[0] - rayStart[0]) * invDirX;
        const t2x = (aabbMax[0] - rayStart[0]) * invDirX;
        const tNearX = Math.min(t1x, t2x);
        const tFarX = Math.max(t1x, t2x);

        const t1y = (aabbMin[1] - rayStart[1]) * invDirY;
        const t2y = (aabbMax[1] - rayStart[1]) * invDirY;
        const tNearY = Math.min(t1y, t2y);
        const tFarY = Math.max(t1y, t2y);

        const t1z = (aabbMin[2] - rayStart[2]) * invDirZ;
        const t2z = (aabbMax[2] - rayStart[2]) * invDirZ;
        const tNearZ = Math.min(t1z, t2z);
        const tFarZ = Math.max(t1z, t2z);

        const tNear = Math.max(tNearX, tNearY, tNearZ);
        const tFar = Math.min(tFarX, tFarY, tFarZ);

        let face: 0 | 1 | 2 | 3 | 4 | 5 = 0; // Default face

        // Check for valid intersection where entry happens before exit and exit is not behind the ray start
        if (tNear < tFar && tFar >= 0) {
          const epsilon = 1e-5; // Tolerance for floating point comparison
          if (Math.abs(tNear - tNearX) < epsilon) {
            face = lookDirection[0] < 0 ? 0 : 1; // Hit +X (0) or -X (1) face
          } else if (Math.abs(tNear - tNearY) < epsilon) {
            face = lookDirection[1] < 0 ? 2 : 3; // Hit +Y (2) or -Y (3) face
          } else {
            // Must be Z face
            face = lookDirection[2] < 0 ? 4 : 5; // Hit +Z (4) or -Z (5) face
          }
        } else {
          // This case should ideally not be reached if the stepped ray found a block.
          // Log a warning and return null as we couldn't reliably determine the face.
          log.warn(
            "Main",
            `Ray-AABB intersection failed for block ${vec3.str(
              block
            )} despite stepped ray hit. tNear: ${tNear.toFixed(
              3
            )}, tFar: ${tFar.toFixed(3)}`
          );
          return null; // Indicate failure to determine face accurately
        }
        // --- End Face Calculation ---

        return {
          block,
          face,
        };
      }

      vec3.copy(lastPos, currentPos);
    }

    return null;
  };

  const physicsStep = async (deltaTimeMs: number) => {
    // --- Physics Update ---
    // Note: updatePhysics is now synchronous, async operations are handled separately
    updatePhysics(
      playerState,
      keyboardState,
      cameraYaw,
      deltaTimeMs,
      chunkDataCache,
      requestChunkData
    );

    if (keyboardState.pressedKeys.has("KeyC")) {
      debugCameraEnabled = !debugCameraEnabled;
    }
    if (keyboardState.pressedKeys.has("KeyV")) {
      debugMode = !debugMode;
    }
    if (keyboardState.pressedKeys.has("KeyB")) {
      // Show chunk storage stats from workers
      workerManager.queueTask({
        type: "getStorageStats",
      });
      log("Main", "Requested storage stats from workers");
    }
    if (keyboardState.pressedKeys.has("KeyF")) {
      // Manually flush pending saves to storage
      workerManager.queueTask({
        type: "flushPendingSaves",
      });
      log("Main", "Requested manual flush of pending saves from workers");
    }
    if (keyboardState.pressedKeys.has("KeyG")) {
      // Check storage status for debugging
      workerManager.queueTask({
        type: "getStorageStatus",
      });
      log("Main", "Requested storage status from workers");
    }
    if (keyboardState.downKeys.has("KeyH")) {
      fov += 0.01;
    }
    if (keyboardState.downKeys.has("KeyL")) {
      fov -= 0.01;
    }

    if (keyboardState.pressedKeys.has("KeyQ")) {
      blockToPlace -= 1;
      if (blockToPlace <= VoxelType.AIR) {
        blockToPlace = VoxelType.REDSTONE;
      }
      updateToolbar();
    }
    if (keyboardState.pressedKeys.has("KeyE")) {
      blockToPlace += 1;
      if (blockToPlace > VoxelType.REDSTONE) {
        blockToPlace = VoxelType.STONE;
      }
      updateToolbar();
    }
    if (keyboardState.mouseDown && blockLookedAt) {
      const { block } = blockLookedAt;
      // Handle block removal asynchronously to avoid blocking the game loop
      (async () => {
        const chunkPosition = getChunkOfPosition(block);
        const chunkData = await requestChunkData(chunkPosition);
        if (!chunkData) return;
        const chunk = new Chunk(chunkPosition, chunkData);
        const localPosition = getLocalPosition(block);
        chunk.setVoxel(localPosition, VoxelType.AIR);
        
        // Update cache
        chunkDataCache.set(getChunkKey(chunkPosition), chunk.data);

        workerManager.queueTask(
          {
            type: "renderChunk",
            position: chunk.position,
            data: chunk.data,
          },
          undefined,
          true
        );
      })();
    } else if (keyboardState.mouseRightClicked && blockLookedAt) {
      const { block, face } = blockLookedAt;
      const newBlock = vec3.clone(block);
      vec3.add(newBlock, block, FACE_NORMALS[face]);
      // Handle block placement asynchronously to avoid blocking the game loop
      (async () => {
        const chunkPosition = getChunkOfPosition(newBlock);
        const localPosition = getLocalPosition(newBlock);
        const chunkData = await requestChunkData(chunkPosition);
        if (!chunkData) return;
        const chunk = new Chunk(chunkPosition, chunkData);
        chunk.setVoxel(localPosition, blockToPlace);
        
        // Update cache
        chunkDataCache.set(getChunkKey(chunkPosition), chunk.data);

        workerManager.queueTask(
          {
            type: "renderChunk",
            position: chunk.position,
            data: chunk.data,
          },
          undefined,
          true
        );
      })();
    }

    keyboardState.pressedKeys.clear();
    keyboardState.mouseClicked = false;
    keyboardState.mouseRightClicked = false;
  };

  let chunksToUnloadQueue: string[] = [];
  let lastUnloadCheck = 0;
  
  // Three-tier unloading system:
  // 1. Storage: Never deleted (permanent)
  // 2. Mesh rendering: Deleted at LOAD_RADIUS + UNLOAD_BUFFER (for performance)
  // 3. Physics cache: Deleted at PHYSICS_CACHE_RADIUS (for memory management)
  const unloadChunks = () => {
    const now = performance.now();
    
    // Only check for new chunks to unload every 5 seconds
    if (now - lastUnloadCheck > 5000) {
      lastUnloadCheck = now;
      const playerChunk = getChunkOfPosition(playerState.position);
      
      // Get all chunk keys and send to worker for processing
      const allChunkKeys = Array.from(rendererState.chunkManager.chunkGeometryInfo.keys());
      
      if (allChunkKeys.length > 0) {
        workerManager.queueTask({
          type: "unloadChunks",
          allChunkKeys,
          playerPosition: playerChunk,
          loadRadiusXZ: LOAD_RADIUS_XZ,
          loadRadiusY: LOAD_RADIUS_Y,
          unloadBufferXZ: UNLOAD_BUFFER_XZ,
          unloadBufferY: UNLOAD_BUFFER_Y
        });
      }
    }

    setTimeout(
      () =>
        requestIdleCallback(unloadChunks, {
          timeout: 2500,
        }),
      100 // Check more frequently for queue processing
    );
  };
  
  // Separate function to process the unload queue in batches
  const processUnloadQueue = () => {
    if (chunksToUnloadQueue.length === 0) {
      requestAnimationFrame(processUnloadQueue);
      return;
    }
    
    const MAX_CHUNKS_PER_FRAME = 500; // Higher limit for faster unloading
    const chunksThisFrame = chunksToUnloadQueue.splice(0, MAX_CHUNKS_PER_FRAME);
    
    for (const key of chunksThisFrame) {
      const chunkInfo = rendererState.chunkManager.chunkGeometryInfo.get(key);
      if (chunkInfo) {
        // Always clean up renderer resources (meshes) when outside unload radius
        // NOTE: This only deletes mesh/geometry data from GPU, NOT from storage
        rendererState.chunkManager.deleteChunk(chunkInfo.position);
        
        // Remove from requested chunks
        requestedChunkKeys.delete(key);
        
        log("Debug", `Unloaded chunk mesh ${key} from renderer`);
      }
    }
    
    if (chunksThisFrame.length > 0) {
      log("Main", `Processed ${chunksThisFrame.length} chunk meshes for unloading`);
    }
    
    requestAnimationFrame(processUnloadQueue);
  };

  // Separate function to clean up physics cache (more aggressive, smaller radius)
  const cleanupPhysicsCache = () => {
    const playerChunk = getChunkOfPosition(playerState.position);
    const chunksToRemove: string[] = [];
    
    // Check each cached chunk to see if it's outside physics radius
    for (const [key, _] of chunkDataCache) {
      const [x, y, z] = key.split(',').map(Number);
      const dx = Math.abs(x - playerChunk[0]);
      const dy = Math.abs(y - playerChunk[1]);
      const dz = Math.abs(z - playerChunk[2]);
      
      // Remove if outside physics cache radius
      if (dx > PHYSICS_CACHE_RADIUS_XZ || dy > PHYSICS_CACHE_RADIUS_Y || dz > PHYSICS_CACHE_RADIUS_XZ) {
        chunksToRemove.push(key);
      }
    }
    
    // Remove chunks outside physics radius
    for (const key of chunksToRemove) {
      chunkDataCache.delete(key);
    }
    
    if (chunksToRemove.length > 0) {
      log("Main", `Cleaned up ${chunksToRemove.length} chunks from physics cache, cache size: ${chunkDataCache.size}`);
    }
  };

  let lastHandledChunkPosition: vec3 | null = null;
  const fn = () => {
    const currentChunk = getChunkOfPosition(playerState.position);
    if (!lastHandledChunkPosition || !vec3.equals(lastHandledChunkPosition, currentChunk)) {
      lastHandledChunkPosition = vec3.clone(currentChunk);

      const playerChunk = getChunkOfPosition(playerState.position);
      log("Main", `Player at position ${playerState.position[0]},${playerState.position[1]},${playerState.position[2]} -> chunk ${playerChunk[0]},${playerChunk[1]},${playerChunk[2]}`);

      let chunksRequested = 0;
      for (let yOffset = -LOAD_RADIUS_Y; yOffset <= LOAD_RADIUS_Y; yOffset++) {
        for (
          let zOffset = -LOAD_RADIUS_XZ;
          zOffset <= LOAD_RADIUS_XZ;
          zOffset++
        ) {
          for (
            let xOffset = -LOAD_RADIUS_XZ;
            xOffset <= LOAD_RADIUS_XZ;
            xOffset++
          ) {
            const chunkPos = vec3.fromValues(
              playerChunk[0] + xOffset,
              playerChunk[1] + yOffset,
              playerChunk[2] + zOffset
            );
            const key = getChunkKey(chunkPos);
            if (!requestedChunkKeys.has(key)) {
              requestedChunkKeys.add(key);
              chunksRequested++;
              workerManager.queueTask({
                type: "requestChunk",
                position: chunkPos,
              });
            }
          }
        }
      }
      if (chunksRequested > 0) {
        log("Main", `Requested ${chunksRequested} new chunks, total requested: ${requestedChunkKeys.size}`);
      }
    }

    setTimeout(
      () => requestIdleCallback(fn, { timeout: 500 }),
      100 // Check more frequently
    );
  };

  // Start chunk loading immediately
  fn();
  
  // Also request chunks around spawn immediately
  const spawnChunk = getChunkOfPosition(playerState.position);
  let initialChunksRequested = 0;
  for (let yOffset = -2; yOffset <= 2; yOffset++) {
    for (let zOffset = -4; zOffset <= 4; zOffset++) {
      for (let xOffset = -4; xOffset <= 4; xOffset++) {
        const chunkPos = vec3.fromValues(
          spawnChunk[0] + xOffset,
          spawnChunk[1] + yOffset,
          spawnChunk[2] + zOffset
        );
        const key = getChunkKey(chunkPos);
        if (!requestedChunkKeys.has(key)) {
          requestedChunkKeys.add(key);
          initialChunksRequested++;
          workerManager.queueTask({
            type: "requestChunk",
            position: chunkPos,
          });
        }
      }
    }
  }
  log("Main", `Requested ${initialChunksRequested} initial chunks around spawn`);

  unloadChunks();
  processUnloadQueue(); // Start the queue processor
  
  // Start physics cache cleanup (more frequent, smaller radius)
  const startPhysicsCacheCleanup = () => {
    cleanupPhysicsCache();
    setTimeout(startPhysicsCacheCleanup, 2000); // Every 2 seconds
  };
  startPhysicsCacheCleanup();

  let debugCameraEnabled = false;
  let fov = Math.PI / 4;
  let blockLookedAt: Awaited<ReturnType<typeof getBlockLookedAt>> | null = null;
  // --- Game Loop Function ---
  let lastTotalTriangles = 0;
  let lastFrameTime = performance.now();
  let physicsTimeMs = 0;
  let renderTimeMs = 0;
  let lookAtTimeMs = 0;
  function frame(deltaTime: number) {
    // console.time("frame"); // Profile the entire frame function
    if (!rendererState) return; // Renderer must be initialized

    const debugCameraPosition = vec3.fromValues(
      playerState.position[0] - 150,
      playerState.position[1] + 150,
      playerState.position[2] - 150
    );
    const debugCameraTarget = vec3.fromValues(
      playerState.position[0],
      playerState.position[1],
      playerState.position[2]
    );

    // console.time("getBlockLookedAt");
    const lookAtStart = performance.now();
    blockLookedAt = getBlockLookedAt(
      playerState.getCameraPosition(),
      cameraYaw
    );
    lookAtTimeMs = performance.now() - lookAtStart;
    // console.timeEnd("getBlockLookedAt");
    const highlightedBlockPositions: vec3[] = [];
    if (blockLookedAt?.block) {
      highlightedBlockPositions.push(blockLookedAt.block);
    }

    // --- Rendering ---
    // console.time("renderFrame");
    const renderStart = performance.now();
    const renderResult = rendererState.renderFrame(
      playerState.getCameraPosition(),
      cameraPitch,
      cameraYaw,
      highlightedBlockPositions,
      fov,
      debugCameraEnabled
        ? {
          position: debugCameraPosition,
          target: debugCameraTarget,
        }
        : undefined,
      debugMode
    );
    renderTimeMs = performance.now() - renderStart;
    // console.timeEnd("renderFrame");
    lastTotalTriangles = renderResult.totalTriangles;

    const playerChunk = getChunkOfPosition(playerState.position);

    // --- Update Debug Info ---
    // console.time("debugInfoUpdate");
    if (debugInfoElement) {
      if (frameTimes.length >= maxFrameSamples) {
        frameTimes.length = maxFrameSamples - 1;
      }
      frameTimes.unshift(deltaTime);
      const avgDelta =
        frameTimes.length > 0
          ? frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length
          : 0;
      const fps = avgDelta > 0 ? 1000 / avgDelta : 0;

      const lookDirection = vec3.create();
      lookDirection[0] = Math.cos(cameraPitch) * Math.sin(cameraYaw);
      lookDirection[1] = Math.sin(cameraPitch);
      lookDirection[2] = Math.cos(cameraPitch) * Math.cos(cameraYaw);
      vec3.normalize(lookDirection, lookDirection);

      const meshingMode = ENABLE_GREEDY_MESHING ? "Greedy" : "Naive";
      
      // Memory usage information
      const getMemoryInfo = () => {
        if ('memory' in performance) {
          const memory = (performance as any).memory;
          const usedJSHeapSize = memory.usedJSHeapSize / 1024 / 1024;
          const totalJSHeapSize = memory.totalJSHeapSize / 1024 / 1024;
          const jsHeapSizeLimit = memory.jsHeapSizeLimit / 1024 / 1024;
          return `${usedJSHeapSize.toFixed(1)}MB / ${totalJSHeapSize.toFixed(1)}MB (${jsHeapSizeLimit.toFixed(0)}MB limit)`;
        }
        return "N/A";
      };

      debugInfoElement.textContent = `
Pos:    (${playerState.position[0].toFixed(
        1
      )}, ${playerState.position[1].toFixed(
        1
      )}, ${playerState.position[2].toFixed(1)})
Chunk:  (${playerChunk[0]}, ${playerChunk[1]}, ${playerChunk[2]})
Look:   (${lookDirection[0].toFixed(2)}, ${lookDirection[1].toFixed(
        2
      )}, ${lookDirection[2].toFixed(2)})
Chunks: ${rendererState.chunkManager.chunkGeometryInfo.size} (${requestedChunkKeys.size
        } req, ${chunkDataCache.size} cached)
Drawn:  ${rendererState.debugInfo.drawnChunks}
Culled: ${rendererState.debugInfo.culledChunks}
Tris:   ${lastTotalTriangles.toLocaleString()}
FPS:    ${fps.toFixed(1)} (${avgDelta.toFixed(2)} ms)
Physics:${physicsTimeMs.toFixed(2)} ms
Render: ${renderTimeMs.toFixed(2)} ms
LookAt: ${lookAtTimeMs.toFixed(2)} ms
Mesh:   ${meshingMode}
RAM:    ${getMemoryInfo()}
Gnd:    ${playerState.isGrounded} VelY: ${playerState.velocity[1].toFixed(2)}
        `.trim();
    }
    // console.timeEnd("debugInfoUpdate");

    // console.timeEnd("frame"); // End profiling the entire frame function
  }

  const frameLoop = async () => {
    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    lastFrameTime = now;

    // Process physics and potentially queue more chunk updates via worker tasks
    const physicsStart = performance.now();
    await physicsStep(deltaTime);
    physicsTimeMs = performance.now() - physicsStart;

    // Render the frame (will only draw 'ready' chunks)
    frame(deltaTime);

    requestAnimationFrame(frameLoop);
  };

  const updateToolbar = () => {
    if (!toolbarElement) return;
    toolbarElement.innerHTML = "";
    const voxelTypes = [
      { name: "Stone", id: VoxelType.STONE },
      { name: "Grass", id: VoxelType.GRASS },
      { name: "Dirt", id: VoxelType.DIRT },
      { name: "Sand", id: VoxelType.SAND },
      { name: "Star", id: VoxelType.STAR },
      { name: "Water", id: VoxelType.WATER },
      { name: "Lava", id: VoxelType.LAVA },
      { name: "Glass", id: VoxelType.GLASS },
      { name: "Iron", id: VoxelType.IRON },
      { name: "Gold", id: VoxelType.GOLD },
      { name: "Diamond", id: VoxelType.DIAMOND },
      { name: "Emerald", id: VoxelType.EMERALD },
      { name: "Lapis Lazuli", id: VoxelType.LAPIS_LAZULI },
      { name: "Redstone", id: VoxelType.REDSTONE },
    ];
    for (const voxelType of voxelTypes) {
      const button = document.createElement("button");
      button.textContent = voxelType.name;
      button.style.border = "none";
      button.style.backgroundColor =
        voxelType.id === blockToPlace ? "gray" : "transparent";
      button.onclick = (e) => {
        e.preventDefault();
        e.stopPropagation();
        blockToPlace = voxelType.id;
        updateToolbar();
      };
      toolbarElement.appendChild(button);
    }
  };

  updateToolbar();

  requestAnimationFrame(frameLoop);

  window.addEventListener("resize", () => {
    rendererState?.resize(window.innerWidth, window.innerHeight);
  });

  // Flush all pending saves before page unload
  window.addEventListener("beforeunload", async (e) => {
    log("Main", "Page unloading - flushing all pending saves to storage");
    
    // Flush pending saves in all workers
    const flushPromises: Promise<void>[] = [];
    for (let i = 0; i < numWorkers; i++) {
      const promise = new Promise<void>((resolve) => {
        const timeout = setTimeout(() => {
          log.warn("Main", `Worker ${i} flush timeout`);
          resolve();
        }, 3000); // 3 second timeout
        
        const handler = (event: MessageEvent) => {
          if (event.data.type === "flushComplete") {
            workerManager.removeMessageHandler(handler);
            clearTimeout(timeout);
            resolve();
          }
        };
        
        workerManager.addMessageHandler(handler);
        workerManager.queueTask({ type: "flushPendingSaves" });
      });
      flushPromises.push(promise);
    }
    
    // Wait for all workers to flush (with timeout)
    try {
      await Promise.all(flushPromises);
      log("Main", "All workers flushed pending saves");
    } catch (error) {
      log.error("Main", "Error flushing pending saves:", error);
    }
  });
}

main().catch((err) => {
  log.error("Main", err);
});
