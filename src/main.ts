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
} from "./common/constants";
import { PlayerState, updatePhysics } from "./physics"; // Import physics
import { Renderer } from "./renderer"; // Import renderer
import {
  Chunk,
  getChunkKey,
  getChunkOfPosition,
  getLocalPosition,
  type ChunkMesh,
} from "./chunk";
import log from "./logger";
import { KeyboardState } from "./keyboard";
import { VoxelType } from "./common/voxel-types";
import { WorkerManager } from "./worker-manager";

const DEBUG_MODE = true;

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
const loadedChunkData = new Map<string, Uint8Array>();
const requestedChunkKeys = new Set<string>();
const playerState = new PlayerState();
let rendererState: Renderer; // Will be initialized later
let keysUpdatedSinceLastSync: string[] = [];

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
    return; // Stop if renderer fails
  }

  // Initialize worker pool
  const numWorkers = navigator.hardwareConcurrency || 4;
  log("Main", `Initializing ${numWorkers} workers...`);

  const workerManager = new WorkerManager(numWorkers);

  // --- Worker Message Handling ---
  const workerMessageHandler = (event: MessageEvent) => {
    const type = event.data.type;
    log("Worker", `Received message type: ${type}`);

    if (type === "chunkDataAvailable") {
      const { position, voxels } = event.data;
      const key = getChunkKey(position);
      // 'voxels' is now a Uint8Array view on the SharedArrayBuffer from the worker
      const sharedVoxelData = voxels as Uint8Array; // Type assertion for clarity
      // voxelData.set(voxels); // This line is no longer needed
      loadedChunkData.set(key, sharedVoxelData);
      requestedChunkKeys.add(key);
    } else if (type === "chunkMeshEmpty") {
      const key = getChunkKey(event.data.position);
      log("Chunk", `Received empty mesh confirmation for chunk ${key}`);
      requestedChunkKeys.add(key);
    } else if (type === "chunkMeshUpdated") {
      if (!rendererState) throw new Error("Renderer not found");

      const {
        position,
        vertices: verticesBuffer,
        indices: indicesBuffer,
      } = event.data as {
        position: vec3;
        vertices: Float32Array;
        indices: Uint32Array;
      };
      if (!rendererState) return; // Guard against renderer not being ready
      const vertices = new Float32Array(verticesBuffer);
      const indices = new Uint32Array(indicesBuffer);

      const minX = position[0] * CHUNK_SIZE_X;
      const minY = position[1] * CHUNK_SIZE_Y;
      const minZ = position[2] * CHUNK_SIZE_Z;
      const maxX = minX + CHUNK_SIZE_X;
      const maxY = minY + CHUNK_SIZE_Y;
      const maxZ = minZ + CHUNK_SIZE_Z;
      const aabb = {
        min: vec3.fromValues(minX, minY, minZ),
        max: vec3.fromValues(maxX, maxY, maxZ),
      };

      const key = getChunkKey(position);

      // Determine if it's an update or a new add based on chunkManager state
      const isUpdate = rendererState.chunkManager.chunkGeometryInfo.has(key);

      try {
        if (isUpdate) {
          rendererState.chunkManager.updateChunkGeometryInfo(
            position,
            vertices,
            vertices.byteLength,
            indices,
            indices.byteLength,
            aabb
          );
        } else {
          rendererState.chunkManager.addChunk(
            position,
            vertices,
            vertices.byteLength,
            indices,
            indices.byteLength,
            aabb
          );
        }
        // Track the key if add/update didn't throw
        keysUpdatedSinceLastSync.push(key);
      } catch (error) {
        log.error("Main", `Error processing mesh update for ${key}:`, error);
      }
    } else {
      log.warn("Main", `Unknown message type from worker: ${type}`);
    }
  };
  workerManager.setMessageHandler(workerMessageHandler);

  // Debug Info Element
  const debugInfoElement = document.getElementById(
    "debug-info"
  ) as HTMLDivElement;

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
    const STEP_SIZE = 0.05; // Size of each step along the ray

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

      // Check if chunk is loaded
      const chunkData = loadedChunkData.get(chunkKey);
      if (!chunkData) continue;

      // Get local coordinates within chunk
      const localPosition = getLocalPosition(block);

      // Get voxel index
      const chunk = new Chunk(blockChunk, chunkData);

      // Check if block exists (non-zero)
      if (chunk.getVoxel(localPosition) !== VoxelType.AIR) {
        // --- Determine which face was hit ---
        // Calculate the center of the block
        const blockCenter = vec3.add(vec3.create(), block, [0.5, 0.5, 0.5]);
        // Vector from block center to the intersection point
        const intersectionVec = vec3.sub(
          vec3.create(),
          currentPos,
          blockCenter
        );
        // Find the axis with the largest absolute component
        const absX = Math.abs(intersectionVec[0]);
        const absY = Math.abs(intersectionVec[1]);
        const absZ = Math.abs(intersectionVec[2]);

        let face: 0 | 1 | 2 | 3 | 4 | 5 = 0; // Default, should always be overwritten

        if (absX >= absY && absX >= absZ) {
          // Hit X face
          face = intersectionVec[0] > 0 ? 0 : 1; // +X or -X
        } else if (absY >= absX && absY >= absZ) {
          // Hit Y face
          face = intersectionVec[1] > 0 ? 2 : 3; // +Y or -Y
        } else {
          // Hit Z face
          face = intersectionVec[2] > 0 ? 4 : 5; // +Z or -Z
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

  const physicsStep = (deltaTimeMs: number) => {
    // --- Physics Update ---
    updatePhysics(
      playerState,
      keyboardState,
      cameraYaw,
      deltaTimeMs,
      loadedChunkData
    );

    if (keyboardState.pressedKeys.has("KeyC")) {
      debugCameraEnabled = !debugCameraEnabled;
    }
    if (keyboardState.downKeys.has("KeyH")) {
      fov += 0.01;
    }
    if (keyboardState.downKeys.has("KeyL")) {
      fov -= 0.01;
    }

    if (keyboardState.mouseDown && blockLookedAt) {
      const { block } = blockLookedAt;
      const chunkData = loadedChunkData.get(
        getChunkKey(getChunkOfPosition(block))
      );
      if (!chunkData) return;
      const chunk = new Chunk(getChunkOfPosition(block), chunkData);
      const localPosition = getLocalPosition(block);
      chunk.setVoxel(localPosition, VoxelType.AIR);
      chunkData.set(chunk.data);

      workerManager.queueTask(
        {
          type: "renderChunk",
          position: chunk.position,
          data: chunk.data,
        },
        undefined,
        true
      );
    } else if (keyboardState.mouseRightClicked && blockLookedAt) {
      const { block, face } = blockLookedAt;
      const newBlock = vec3.clone(block);
      vec3.add(newBlock, block, FACE_NORMALS[face]);
      const localPosition = getLocalPosition(newBlock);
      const chunkData = loadedChunkData.get(
        getChunkKey(getChunkOfPosition(newBlock))
      );
      if (!chunkData) return;
      const chunk = new Chunk(getChunkOfPosition(newBlock), chunkData);
      chunk.setVoxel(localPosition, VoxelType.STONE);
      chunkData.set(chunk.data);

      workerManager.queueTask(
        {
          type: "renderChunk",
          position: chunk.position,
          data: chunk.data,
        },
        undefined,
        true
      );
    }

    keyboardState.pressedKeys.clear();
    keyboardState.mouseClicked = false;
    keyboardState.mouseRightClicked = false;
  };

  const unloadChunks = () => {
    const playerChunk = getChunkOfPosition(playerState.position);

    for (const [
      key,
      chunkMesh,
    ] of rendererState.chunkManager.chunkGeometryInfo.entries()) {
      const dx = Math.abs(chunkMesh.position[0] - playerChunk[0]);
      const dy = Math.abs(chunkMesh.position[1] - playerChunk[1]);
      const dz = Math.abs(chunkMesh.position[2] - playerChunk[2]);
      if (
        dx > LOAD_RADIUS_XZ + UNLOAD_BUFFER_XZ ||
        dy > LOAD_RADIUS_Y + UNLOAD_BUFFER_Y ||
        dz > LOAD_RADIUS_XZ + UNLOAD_BUFFER_XZ
      ) {
        rendererState.chunkManager.deleteChunk(chunkMesh.position);
        loadedChunkData.delete(key);
        requestedChunkKeys.delete(key);
      }
    }

    setTimeout(
      () =>
        requestIdleCallback(unloadChunks, {
          timeout: 2500,
        }),
      5000
    );
  };

  unloadChunks();

  let debugCameraEnabled = false;
  let fov = Math.PI / 4;
  let blockLookedAt: ReturnType<typeof getBlockLookedAt> | null = null;
  // --- Game Loop Function ---
  let lastTotalTriangles = 0;
  function frame(deltaTime: number) {
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

    blockLookedAt = getBlockLookedAt(
      playerState.getCameraPosition(),
      cameraYaw
    );
    const highlightedBlockPositions: vec3[] = [];
    if (blockLookedAt?.block) {
      highlightedBlockPositions.push(blockLookedAt.block);
    }

    // --- Rendering ---
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
      DEBUG_MODE
    );
    lastTotalTriangles = renderResult.totalTriangles;

    const playerChunk = getChunkOfPosition(playerState.position);

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
            log("Chunk", `Requesting chunk: ${key}`);
            requestedChunkKeys.add(key);
            workerManager.queueTask({
              type: "requestChunk",
              position: chunkPos,
            });
          }
        }
      }
    }

    // --- Update Debug Info ---
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
Chunks: ${rendererState.chunkManager.chunkGeometryInfo.size} (${
        requestedChunkKeys.size
      } req)
Drawn:  ${rendererState.debugInfo.drawnChunks}
Culled: ${rendererState.debugInfo.culledChunks}
Tris:   ${lastTotalTriangles.toLocaleString()}
FPS:    ${fps.toFixed(1)}
Mesh:   ${meshingMode}
Gnd:    ${playerState.isGrounded} VelY: ${playerState.velocity[1].toFixed(2)}
        `.trim();
    }
  }

  let lastFrameTime = performance.now();
  const frameLoop = () => {
    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    lastFrameTime = now;

    // Process physics and potentially queue more chunk updates via worker tasks
    physicsStep(deltaTime);

    // Perform GPU synchronization *before* rendering
    finalizeChunkUpdates();

    // Render the frame (will only draw 'ready' chunks)
    frame(deltaTime);

    requestAnimationFrame(frameLoop);
  };

  requestAnimationFrame(frameLoop);

  window.addEventListener("resize", () => {
    rendererState?.resize(window.innerWidth, window.innerHeight);
  });

  // --- Synchronization Function ---
  function finalizeChunkUpdates() {
    if (keysUpdatedSinceLastSync.length > 0) {
      const keysToMarkReady = [...keysUpdatedSinceLastSync];
      keysUpdatedSinceLastSync = [];

      rendererState.chunkManager.device.queue
        .onSubmittedWorkDone()
        .then(() => {
          for (const key of keysToMarkReady) {
            const chunkInfo =
              rendererState.chunkManager.chunkGeometryInfo.get(key);
            if (chunkInfo) {
              if (chunkInfo.status === "updating") {
                chunkInfo.status = "ready"; // <-- THE CRITICAL STEP
              } else {
                // This can happen if the chunk was deleted or updated again
                // before the promise resolved. Usually safe to ignore.
                // console.warn(
                //   `Sync: Chunk ${key} status was not 'updating' (was ${chunkInfo.status}) when marking ready.`
                // );
              }
            } else {
              // This can happen if the chunk was deleted before the promise resolved.
              console.warn(`Sync: Chunk ${key} not found when marking ready.`);
            }
          }
        })
        .catch((err) => {
          console.error("Sync: onSubmittedWorkDone error:", err);
          // Consider how to handle this - maybe retry marking later?
        });
    }
  }
}

main().catch((err) => {
  log.error("Main", err);
});
