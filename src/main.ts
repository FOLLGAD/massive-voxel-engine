/// <reference types="bun-types" />
/// <reference types="@webgpu/types" />

import { vec3 } from "gl-matrix"; // Keep gl-matrix for look direction vector
import { ENABLE_GREEDY_MESHING } from "./config";
import {
  CHUNK_CONFIG,
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
import { ChunkStorageBase, IdbChunkStorage, IdbWorldsStorage } from './chunk-storage';

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
let renderer: Renderer;
let chunksReceived = 0;
let chunkStorage: ChunkStorageBase;

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
    renderer = await Renderer.create(canvas);
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

  console.log("Loading worldsStorage...");
  const worldsStorage = new IdbWorldsStorage();
  console.log("Loading world...");
  chunkStorage = await worldsStorage.createWorld("sleepyville", 321321312, "pebbletown2");

  const { worldName, worldSeed } = await chunkStorage.ensureInitialized();

  console.log("Ensuring initialized...");
  await chunkStorage.ensureInitialized();
  console.log("Initialized");

  const workerManager = new WorkerManager(numWorkers, { type: "init", worldName, worldSeed });

  // Function to request chunk data from workers when needed for physics
  const requestChunkData = async (posBatch: vec3[]) => {
    const missingPos: vec3[] = [];

    // Check cache first
    for (const position of posBatch) {
      const key = getChunkKey(position);
      if (!chunkDataCache.has(key)) {
        missingPos.push(vec3.clone(position));
      }
    }

    const storedData = await chunkStorage.loadChunks(missingPos);
    for (const [key, data] of storedData) {
      chunkDataCache.set(key, data);
    }
  };

  let chunksToUnloadQueue: string[] = [];
  let currentEpoch = 0;
  let inFlightChunkKeys = new Set<string>();

  // --- Worker Message Handling ---
  const workerMessageHandler = (event: MessageEvent) => {
    const { type, ...data } = event.data;

    if (type === "chunkMeshUpdated") {
      const { position, vertices, indices, visibilityBits, epoch } = data as { position: vec3; vertices: ArrayBuffer; indices: ArrayBuffer; visibilityBits: number; epoch?: number };
      // Always clear in-flight state for this key, even if stale
      const key = getChunkKey(position);
      inFlightChunkKeys.delete(key);
      if (typeof epoch === 'number' && epoch < currentEpoch) {
        // Stale result; ignore applying to renderer
        return;
      }
      const aabb = createAABB(
        vec3.fromValues(position[0] * CHUNK_CONFIG.size.x, position[1] * CHUNK_CONFIG.size.y, position[2] * CHUNK_CONFIG.size.z),
        vec3.fromValues((position[0] + 1) * CHUNK_CONFIG.size.x, (position[1] + 1) * CHUNK_CONFIG.size.y, (position[2] + 1) * CHUNK_CONFIG.size.z)
      );
      renderer.chunkManager.updateChunkGeometryInfo(
        position,
        new Float32Array(vertices),
        vertices.byteLength,
        new Uint32Array(indices),
        indices.byteLength,
        aabb,
        visibilityBits
      );
      // Mark as completed (already removed above)
    } else if (type === "chunkDataAvailable") {
      const { position, voxels, epoch } = data as { position: vec3; voxels: ArrayBuffer; epoch?: number };
      if (typeof epoch === 'number' && epoch < currentEpoch) {
        return;
      }
      const key = getChunkKey(position);
      const voxelData = new Uint8Array(voxels);
      chunkDataCache.set(key, voxelData);
      chunksReceived++;
    } else if (type === "chunkGenerated") {
      const { position, voxels, epoch } = data as { position: vec3; voxels: ArrayBuffer; epoch?: number };
      if (typeof epoch === 'number' && epoch < currentEpoch) {
        return;
      }
      chunkStorage.saveChunk(position, new Uint8Array(voxels));
    } else if (type === "chunkNeedsDeletion") {
      chunkStorage.deleteChunk(data.position);
    } else if (type === "chunksToUnload") {
      chunksToUnloadQueue.push(...data.chunks);
    } else {
      log.warn("Main", `Unknown message type from worker: ${type}`);
    }
  };
  workerManager.setMessageHandler(workerMessageHandler);

  const debugInfoElement = document.getElementById(
    "debug-info"
  ) as HTMLDivElement;

  const toolbarElement = document.getElementById("toolbar") as HTMLDivElement;
  let blockToPlace: VoxelType = VoxelType.STONE;

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

    // Step along the ray
    for (let distance = 0; distance <= MAX_DISTANCE; distance += STEP_SIZE) {
      vec3.scaleAndAdd(currentPos, rayStart, lookDirection, distance);

      // Get block at current position
      const block = vec3.fromValues(
        Math.floor(currentPos[0]),
        Math.floor(currentPos[1]),
        Math.floor(currentPos[2])
      );

      const blockChunk = getChunkOfPosition(block);
      const key = getChunkKey(blockChunk);
      const cached = chunkDataCache.get(key);
      if (!cached) continue;

      const localPosition = getLocalPosition(block);
      const chunk = new Chunk(blockChunk, cached);

      if (chunk.getVoxel(localPosition) !== VoxelType.AIR) {
        // --- Determine which face was hit using Ray-AABB intersection ---
        const aabbMin = block;
        const aabbMax = vec3.add(vec3.create(), block, [1, 1, 1]);

        // Avoid division by zero if lookDirection component is zero
        const invDirX = lookDirection[0] === 0 ? Number.POSITIVE_INFINITY : 1.0 / lookDirection[0];
        const invDirY = lookDirection[1] === 0 ? Number.POSITIVE_INFINITY : 1.0 / lookDirection[1];
        const invDirZ = lookDirection[2] === 0 ? Number.POSITIVE_INFINITY : 1.0 / lookDirection[2];

        const tx1 = (aabbMin[0] - rayStart[0]) * invDirX;
        const tx2 = (aabbMax[0] - rayStart[0]) * invDirX;
        const tminX = Math.min(tx1, tx2);
        const tmaxX = Math.max(tx1, tx2);

        const ty1 = (aabbMin[1] - rayStart[1]) * invDirY;
        const ty2 = (aabbMax[1] - rayStart[1]) * invDirY;
        const tminY = Math.min(ty1, ty2);
        const tmaxY = Math.max(ty1, ty2);

        const tz1 = (aabbMin[2] - rayStart[2]) * invDirZ;
        const tz2 = (aabbMax[2] - rayStart[2]) * invDirZ;
        const tminZ = Math.min(tz1, tz2);
        const tmaxZ = Math.max(tz1, tz2);

        const tmin = Math.max(tminX, tminY, tminZ);
        const tmax = Math.min(tmaxX, tmaxY, tmaxZ);

        if (tmin >= tmax || tmax < 0) {
          return null;
        }

        let face: 0 | 1 | 2 | 3 | 4 | 5 = 0;
        const epsilon = 1e-5;

        if (Math.abs(tmin - tminX) < epsilon) {
          face = lookDirection[0] > 0 ? 1 : 0;
        } else if (Math.abs(tmin - tminY) < epsilon) {
          face = lookDirection[1] > 0 ? 3 : 2;
        } else {
          face = lookDirection[2] > 0 ? 5 : 4;
        }

        return { block, face };
      }
    }

    return null;
  };

  const physicsStep = async (deltaTimeMs: number) => {
    const chunksToLoad = new Set<vec3>();
    const chunkLoadFn = (position: vec3) => {
      if (!chunksToLoad.has(position)) {
        chunksToLoad.add(position);
      }
    };
    updatePhysics(
      playerState,
      keyboardState,
      cameraYaw,
      deltaTimeMs,
      chunkDataCache,
      chunkLoadFn
    );

    requestChunkData(Array.from(chunksToLoad));

    if (keyboardState.pressedKeys.has("KeyC")) {
      debugCameraEnabled = !debugCameraEnabled;
    }
    if (keyboardState.pressedKeys.has("KeyV")) {
      debugMode = !debugMode;
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
    if (keyboardState.pressedKeys.has("KeyJ")) {
      CHUNK_CONFIG.loadRadius.xz -= 1;
      CHUNK_CONFIG.loadRadius.y -= 1;
    }
    if (keyboardState.pressedKeys.has("KeyK")) {
      CHUNK_CONFIG.loadRadius.xz += 1;
      CHUNK_CONFIG.loadRadius.y += 1;
    }

    if (keyboardState.mouseDown && blockLookedAt) {
      const { block } = blockLookedAt;
      const chunkPosition = getChunkOfPosition(block);
      requestChunkData([chunkPosition]);
      const chunkData = chunkDataCache.get(getChunkKey(chunkPosition));

      if (chunkData) {
        const chunk = new Chunk(chunkPosition, chunkData);
        const localPosition = getLocalPosition(block);
        chunk.setVoxel(localPosition, VoxelType.AIR);

        const mainThreadData = chunk.data.slice();
        chunkDataCache.set(getChunkKey(chunkPosition), mainThreadData);
        chunkStorage.saveChunk(chunkPosition, mainThreadData, true);

        const workerData = mainThreadData.slice();
        workerManager.queueTask(
          {
            type: "renderChunk",
            position: chunk.position,
            data: workerData.buffer,
            epoch: currentEpoch,
          },
          [workerData.buffer],
          true
        );
      }
    } else if (keyboardState.mouseRightClicked && blockLookedAt) {
      const { block, face } = blockLookedAt;
      const newBlock = vec3.clone(block);
      vec3.add(newBlock, block, FACE_NORMALS[face]);
      const chunkPosition = getChunkOfPosition(newBlock);
      requestChunkData([chunkPosition]);
      const chunkData = chunkDataCache.get(getChunkKey(chunkPosition));
      if (chunkData) {
        const chunk = new Chunk(chunkPosition, chunkData);
        const localPosition = getLocalPosition(newBlock);
        chunk.setVoxel(localPosition, blockToPlace);

        const mainThreadData = chunk.data.slice();
        chunkDataCache.set(getChunkKey(chunkPosition), mainThreadData);
        chunkStorage.saveChunk(chunkPosition, mainThreadData, true);

        const workerData = mainThreadData.slice();
        workerManager.queueTask(
          {
            type: "renderChunk",
            position: chunk.position,
            data: workerData.buffer,
            epoch: currentEpoch,
          },
          [workerData.buffer],
          true
        );
      }
    }

    keyboardState.pressedKeys.clear();
    keyboardState.mouseClicked = false;
    keyboardState.mouseRightClicked = false;
  };

  let lastUnloadCheck = 0;
  const unloadChunks = () => {
    const now = performance.now();

    if (now - lastUnloadCheck > 5000) {
      lastUnloadCheck = now;
      const playerChunk = getChunkOfPosition(playerState.position);

      const allChunkKeys = Array.from(renderer.chunkManager.chunkGeometryInfo.keys());

      if (allChunkKeys.length > 0) {
        workerManager.queueTask({
          type: "unloadChunks",
          allChunkKeys,
          playerPosition: playerChunk,
          loadRadiusXZ: CHUNK_CONFIG.loadRadius.xz,
          loadRadiusY: CHUNK_CONFIG.loadRadius.y,
          unloadBufferXZ: UNLOAD_BUFFER_XZ,
          unloadBufferY: UNLOAD_BUFFER_Y
        }, undefined, true);
      }

      for (const [chunkKey, chunkInfo] of renderer.chunkManager.chunkGeometryInfo.entries()) {
        const chunkPos = chunkInfo.position;

        const dx = Math.abs(chunkPos[0] - playerChunk[0]);
        const dy = Math.abs(chunkPos[1] - playerChunk[1]);
        const dz = Math.abs(chunkPos[2] - playerChunk[2]);

        if (dx > CHUNK_CONFIG.loadRadius.xz + UNLOAD_BUFFER_XZ || dy > CHUNK_CONFIG.loadRadius.y + UNLOAD_BUFFER_Y || dz > CHUNK_CONFIG.loadRadius.xz + UNLOAD_BUFFER_XZ) {
          chunksToUnloadQueue.push(chunkKey);
        }
      }
    }

    setTimeout(() => requestIdleCallback(() => unloadChunks(), { timeout: 100 }), 100);
  };

  const processUnloadQueue = () => {
    const chunksThisFrame = chunksToUnloadQueue;
    chunksToUnloadQueue = [];

    for (const key of chunksThisFrame) {
      const chunkInfo = renderer.chunkManager.chunkGeometryInfo.get(key);
      if (chunkInfo) {
        renderer.chunkManager.deleteChunk(chunkInfo.position);
        requestedChunkKeys.delete(key);
      }
    }
    setTimeout(() => requestIdleCallback(processUnloadQueue, { timeout: 500 }), 500);
  }

  const cleanupPhysicsCache = () => {
    const playerChunk = getChunkOfPosition(playerState.position);
    for (const [key, _] of chunkDataCache) {
      const [x, y, z] = key.split(',').map(Number);
      if (Math.abs(x - playerChunk[0]) > PHYSICS_CACHE_RADIUS_XZ || Math.abs(y - playerChunk[1]) > PHYSICS_CACHE_RADIUS_Y || Math.abs(z - playerChunk[2]) > PHYSICS_CACHE_RADIUS_XZ) {
        chunkDataCache.delete(key);
      }
    }
  };

  setInterval(cleanupPhysicsCache, 2000);

  let lastHandledChunkPosition: vec3 | null = null;
  const loadChunksNearby = async () => {
    const playerChunk = getChunkOfPosition(playerState.position);
    // Increment epoch when entering a new player chunk
    if (!lastHandledChunkPosition || !vec3.equals(lastHandledChunkPosition, playerChunk)) {
      lastHandledChunkPosition = vec3.clone(playerChunk);
      currentEpoch += 1;
    }

    // Build candidate chunk positions within radius
    const candidatePositions: vec3[] = [];
    for (let yOffset = -CHUNK_CONFIG.loadRadius.y; yOffset <= CHUNK_CONFIG.loadRadius.y; yOffset++) {
      for (let zOffset = -CHUNK_CONFIG.loadRadius.xz; zOffset <= CHUNK_CONFIG.loadRadius.xz; zOffset++) {
        for (let xOffset = -CHUNK_CONFIG.loadRadius.xz; xOffset <= CHUNK_CONFIG.loadRadius.xz; xOffset++) {
          const pos = vec3.fromValues(
            playerChunk[0] + xOffset,
            playerChunk[1] + yOffset,
            playerChunk[2] + zOffset
          );
          candidatePositions.push(pos);
        }
      }
    }

    // Compute priority scores (distance + view bias)
    const camPos = playerState.getCameraPosition();
    const lookDirection = vec3.fromValues(
      Math.cos(cameraPitch) * Math.sin(cameraYaw),
      Math.sin(cameraPitch),
      Math.cos(cameraPitch) * Math.cos(cameraYaw)
    );
    vec3.normalize(lookDirection, lookDirection);

    const scored: Array<{ position: vec3; score: number; key: string }> = [];
    for (const pos of candidatePositions) {
      const key = getChunkKey(pos);
      // Skip if already rendered or currently in flight
      if (renderer.chunkManager.chunkGeometryInfo.has(key)) continue;
      if (inFlightChunkKeys.has(key)) continue;

      const dx = pos[0] - playerChunk[0];
      const dy = pos[1] - playerChunk[1];
      const dz = pos[2] - playerChunk[2];
      const manhattan = Math.abs(dx) + Math.abs(dy) + Math.abs(dz);

      // View direction bias using vector from camera to chunk center
      const chunkWorldCenter = vec3.fromValues(
        (pos[0] + 0.5) * CHUNK_CONFIG.size.x,
        (pos[1] + 0.5) * CHUNK_CONFIG.size.y,
        (pos[2] + 0.5) * CHUNK_CONFIG.size.z,
      );
      const toChunk = vec3.subtract(vec3.create(), chunkWorldCenter, camPos);
      vec3.normalize(toChunk, toChunk);
      const facing = Math.max(0, vec3.dot(lookDirection, toChunk)); // 0..1
      const viewPenalty = 1 - facing; // 0 if directly facing

      const verticalPenalty = Math.abs(dy) * 0.2;

      const score = manhattan + viewPenalty * 2 + verticalPenalty;
      scored.push({ position: pos, score, key });
    }

    // Sort by score ascending
    scored.sort((a, b) => a.score - b.score);

    // Select a budget to schedule this tick
    const BUDGET = Math.max(8, (navigator.hardwareConcurrency || 4) * 8);
    const toSchedule = scored.slice(0, BUDGET).map(s => s.position);

    if (toSchedule.length > 0) {
      // Load any stored voxel data for these
      const storedChunks = await chunkStorage.loadChunks(toSchedule);

      for (const pos of toSchedule) {
        const key = getChunkKey(pos);
        if (inFlightChunkKeys.has(key)) continue;
        const storedData = storedChunks.get(key);
        inFlightChunkKeys.add(key);
        requestedChunkKeys.add(key);
        workerManager.queueTask({
          type: "requestChunk",
          position: pos,
          data: storedData ? storedData.buffer : null,
          epoch: currentEpoch,
        }, storedData ? [storedData.buffer] : []);
      }
    }

    setTimeout(() => requestIdleCallback(loadChunksNearby, { timeout: 200 }), 200);
  };

  loadChunksNearby();
  unloadChunks();
  processUnloadQueue();

  let debugCameraEnabled = false;
  let fov = Math.PI / 4;
  let blockLookedAt: Awaited<ReturnType<typeof getBlockLookedAt>> | null = null;
  let lastFrameTime = performance.now();
  let physicsTimeMs = 0;
  let renderTimeMs = 0;
  let lookAtTimeMs = 0;

  function frame(deltaTime: number) {
    if (!renderer) return;

    const lookAtStart = performance.now();
    blockLookedAt = getBlockLookedAt(
      playerState.getCameraPosition(),
      cameraYaw
    );
    lookAtTimeMs = performance.now() - lookAtStart;
    const highlightedBlockPositions: vec3[] = [];
    if (blockLookedAt?.block) {
      highlightedBlockPositions.push(blockLookedAt.block);
    }

    const renderStart = performance.now();
    renderer.renderFrame(
      playerState.getCameraPosition(),
      cameraPitch,
      cameraYaw,
      highlightedBlockPositions,
      fov,
      debugCameraEnabled ? {
        position: vec3.fromValues(playerState.position[0] - 150, playerState.position[1] + 150, playerState.position[2] - 150),
        target: playerState.position,
      } : undefined,
      debugMode
    );
    renderTimeMs = performance.now() - renderStart;

    if (debugInfoElement) {
      if (frameTimes.length >= maxFrameSamples) {
        frameTimes.shift();
      }
      frameTimes.push(deltaTime);
      const avgDelta = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
      const fps = 1000 / avgDelta;
      const playerChunk = getChunkOfPosition(playerState.position);
      const lookDirection = vec3.fromValues(Math.cos(cameraPitch) * Math.sin(cameraYaw), Math.sin(cameraPitch), Math.cos(cameraPitch) * Math.cos(cameraYaw));

      debugInfoElement.textContent = `
Pos:    (${playerState.position[0].toFixed(1)}, ${playerState.position[1].toFixed(1)}, ${playerState.position[2].toFixed(1)})
Chunk:  (${playerChunk[0]}, ${playerChunk[1]}, ${playerChunk[2]})
Look:   (${lookDirection[0].toFixed(2)}, ${lookDirection[1].toFixed(2)}, ${lookDirection[2].toFixed(2)})
Chunks: ${renderer.chunkManager.chunkGeometryInfo.size} (${requestedChunkKeys.size} req, ${chunkDataCache.size} cached)
FPS:    ${fps.toFixed(1)} (${avgDelta.toFixed(2)} ms)
Physics:${physicsTimeMs.toFixed(2)} ms
Render: ${renderTimeMs.toFixed(2)} ms
LookAt: ${lookAtTimeMs.toFixed(2)} ms
Mesh:   ${ENABLE_GREEDY_MESHING ? "Greedy" : "Naive"}
Gnd:    ${playerState.isGrounded} VelY: ${playerState.velocity[1].toFixed(2)}
Chunks: ${CHUNK_CONFIG.loadRadius.xz}x${CHUNK_CONFIG.loadRadius.y}
      `.trim();
    }
  }

  const frameLoop = async () => {
    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    lastFrameTime = now;

    const physicsStart = performance.now();
    await physicsStep(deltaTime);
    physicsTimeMs = performance.now() - physicsStart;

    frame(deltaTime);

    requestAnimationFrame(frameLoop);
  };

  const updateToolbar = () => {
    if (!toolbarElement) return;
    toolbarElement.innerHTML = "";
    const voxelTypes = Object.entries(VoxelType).filter(([key]) => isNaN(Number(key)) && key !== 'AIR').map(([name, id]) => ({ name, id: id as VoxelType }));

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
    renderer?.resize(window.innerWidth, window.innerHeight);
  });

  window.addEventListener("beforeunload", async (e) => {
    log("Main", "Page unloading - flushing all pending saves to storage");
    await chunkStorage.flushPendingSaves();
    log("Main", "All pending saves flushed");
  });
}

main().catch((err) => {
  log.error("Main", err);
});
