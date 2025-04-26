/// <reference types="bun-types" />
/// <reference types="@webgpu/types" />

import { vec3 } from "gl-matrix"; // Keep gl-matrix for look direction vector
import { ENABLE_GREEDY_MESHING } from "./config";
import {
  CHUNK_SIZE_X,
  CHUNK_SIZE_Y,
  CHUNK_SIZE_Z,
  LOAD_RADIUS_XZ,
  LOAD_RADIUS_Y,
  UNLOAD_BUFFER_XZ,
  UNLOAD_BUFFER_Y,
} from "./common/constants";
import { PlayerState, updatePhysics } from "./physics"; // Import physics
import { Renderer } from "./renderer"; // Import renderer
import { getChunkKey, type ChunkMesh } from "./chunk";
import log from "./logger";
import { KeyboardState } from "./keyboard";

const DEBUG_MODE = false;

log("Main", "Main script loaded.");

// --- Global State ---
const chunkMeshes = new Map<string, ChunkMesh>();
const loadedChunkData = new Map<string, Uint8Array>();
const requestedChunkKeys = new Set<string>();
const playerState = new PlayerState();
let rendererState: Renderer | null = null; // Will be initialized later

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
      console.log("The pointer lock status is now locked");
      document.addEventListener("mousemove", updatePosition, false);
    } else {
      console.log("The pointer lock status is now unlocked");
      document.removeEventListener("mousemove", updatePosition, false);
    }
  });

  window.addEventListener("keydown", (e) => {
    console.log("keydown", e.code);
    if (e.repeat) return;
    keyboardState.pressedKeys.add(e.code);
    keyboardState.downKeys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    console.log("keyup", e.code);
    keyboardState.downKeys.delete(e.code);
  });
  window.addEventListener("mousedown", () => {
    keyboardState.mouseDown = true;
    keyboardState.mouseClicked = true;
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
  const workers: Worker[] = [];
  for (let i = 0; i < numWorkers; i++) {
    workers.push(new Worker("./worker.js", { type: "module" }));
  }
  let nextWorkerIndex = 0;
  log("Main", `${numWorkers} Workers initialized`);

  // --- Worker Message Handling ---
  const workerMessageHandler = (event: MessageEvent) => {
    const type = event.data.type;
    log("Worker", `Received message type: ${type}`);

    if (type === "chunkMeshAvailable") {
      const {
        position,
        vertices: verticesBuffer,
        indices: indicesBuffer,
      } = event.data;
      if (!rendererState) return; // Guard against renderer not being ready
      const vertices = new Float32Array(verticesBuffer);
      const indices = new Uint32Array(indicesBuffer);
      try {
        const vertexBuffer = rendererState.device.createBuffer({
          label: `chunk-${position.x}-${position.y}-${position.z}-vertex`,
          size: vertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        rendererState.device.queue.writeBuffer(vertexBuffer, 0, vertices);

        const indexBuffer = rendererState.device.createBuffer({
          label: `chunk-${position.x}-${position.y}-${position.z}-index`,
          size: indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        rendererState.device.queue.writeBuffer(indexBuffer, 0, indices);

        const key = getChunkKey(position);

        // Calculate AABB in world coordinates
        const minX = position.x * CHUNK_SIZE_X;
        const minY = position.y * CHUNK_SIZE_Y;
        const minZ = position.z * CHUNK_SIZE_Z;
        const maxX = minX + CHUNK_SIZE_X;
        const maxY = minY + CHUNK_SIZE_Y;
        const maxZ = minZ + CHUNK_SIZE_Z;

        chunkMeshes.set(key, {
          position: position,
          vertexBuffer: vertexBuffer,
          indexBuffer: indexBuffer,
          indexCount: indices.length,
          aabb: {
            min: vec3.fromValues(minX, minY, minZ),
            max: vec3.fromValues(maxX, maxY, maxZ),
          },
        });
        requestedChunkKeys.add(key);
      } catch (error) {
        log.error("Main", "Error creating/writing GPU buffers:", error);
      }
    } else if (type === "chunkDataAvailable") {
      const { position, voxels } = event.data;
      const key = getChunkKey(position);
      loadedChunkData.set(key, new Uint8Array(voxels));
      requestedChunkKeys.add(key);
    } else if (type === "chunkMeshEmpty") {
      const key = getChunkKey(event.data.position);
      log("Chunk", `Received empty mesh confirmation for chunk ${key}`);
      requestedChunkKeys.add(key);
    } else {
      log.warn("Main", `Unknown message type from worker: ${type}`);
    }
  };
  for (const worker of workers) {
    worker.onmessage = workerMessageHandler;
  }

  // Debug Info Element
  const debugInfoElement = document.getElementById(
    "debug-info"
  ) as HTMLDivElement;

  const getBlockLookedAt = (position: vec3, cameraYaw: number) => {
    const lookDirection = vec3.create();
    lookDirection[0] = Math.cos(cameraPitch) * Math.sin(cameraYaw);
    lookDirection[1] = Math.sin(cameraPitch);
    lookDirection[2] = Math.cos(cameraPitch) * Math.cos(cameraYaw);
    vec3.normalize(lookDirection, lookDirection);

    const rayStart = vec3.create();
    vec3.copy(rayStart, position);
    const rayEnd = vec3.create();
    vec3.scaleAndAdd(rayEnd, rayStart, lookDirection, 100);

    // Raycast to find the block the player is looking at
    const MAX_DISTANCE = 20.0; // Maximum distance to check for blocks
    const STEP_SIZE = 0.05; // Size of each step along the ray

    // Start from eye position (slightly above player position)
    const eyePosition = vec3.clone(rayStart);
    eyePosition[1] += 1.7; // Approximate eye height

    const currentPos = vec3.clone(eyePosition);
    let hitBlock: { x: number; y: number; z: number; face: number } | null =
      null;
    let hitFace: number | null = null;
    const lastPos = vec3.clone(currentPos);

    // Step along the ray
    for (let distance = 0; distance <= MAX_DISTANCE; distance += STEP_SIZE) {
      vec3.scaleAndAdd(currentPos, eyePosition, lookDirection, distance);

      // Get block at current position
      const blockX = Math.floor(currentPos[0]);
      const blockY = Math.floor(currentPos[1]);
      const blockZ = Math.floor(currentPos[2]);

      // Get chunk key for this block
      const chunkX = Math.floor(blockX / CHUNK_SIZE_X);
      const chunkY = Math.floor(blockY / CHUNK_SIZE_Y);
      const chunkZ = Math.floor(blockZ / CHUNK_SIZE_Z);
      const chunkKey = `${chunkX},${chunkY},${chunkZ}`;

      // Check if chunk is loaded
      const chunkData = loadedChunkData.get(chunkKey);
      if (!chunkData) continue;

      // Get local coordinates within chunk
      const localX = ((blockX % CHUNK_SIZE_X) + CHUNK_SIZE_X) % CHUNK_SIZE_X;
      const localY = ((blockY % CHUNK_SIZE_Y) + CHUNK_SIZE_Y) % CHUNK_SIZE_Y;
      const localZ = ((blockZ % CHUNK_SIZE_Z) + CHUNK_SIZE_Z) % CHUNK_SIZE_Z;

      // Get voxel index
      const voxelIndex =
        localX + localY * CHUNK_SIZE_X + localZ * CHUNK_SIZE_X * CHUNK_SIZE_Y;

      // Check if block exists (non-zero)
      if (chunkData[voxelIndex] !== 0) {
        // Determine which face was hit by checking which axis changed most recently
        const dx = currentPos[0] - lastPos[0];
        const dy = currentPos[1] - lastPos[1];
        const dz = currentPos[2] - lastPos[2];

        const absDx = Math.abs(dx);
        const absDy = Math.abs(dy);
        const absDz = Math.abs(dz);

        if (absDx >= absDy && absDx >= absDz) {
          hitFace = dx > 0 ? 0 : 1; // +X or -X face
        } else if (absDy >= absDx && absDy >= absDz) {
          hitFace = dy > 0 ? 2 : 3; // +Y or -Y face
        } else {
          hitFace = dz > 0 ? 4 : 5; // +Z or -Z face
        }

        hitBlock = { x: blockX, y: blockY, z: blockZ, face: hitFace };
        break;
      }

      vec3.copy(lastPos, currentPos);
    }

    return hitBlock;
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

    if (keyboardState.mouseClicked) {
      const blockLookedAt = getBlockLookedAt(playerState.position, cameraYaw);
      if (blockLookedAt) {
        log(
          "Main",
          `Block looked at: ${blockLookedAt.x}, ${blockLookedAt.y}, ${blockLookedAt.z}, ${blockLookedAt.face}`
        );
      } else {
        log("Main", "No block looked at");
      }
    }

    keyboardState.pressedKeys.clear();
    keyboardState.mouseClicked = false;
  };

  const unloadChunks = () => {
    const playerChunkX = Math.floor(playerState.position[0] / CHUNK_SIZE_X);
    const playerChunkY = Math.floor(playerState.position[1] / CHUNK_SIZE_Y);
    const playerChunkZ = Math.floor(playerState.position[2] / CHUNK_SIZE_Z);

    for (const [key, chunkMesh] of chunkMeshes.entries()) {
      const { x, y, z } = chunkMesh.position;
      const dx = Math.abs(x - playerChunkX);
      const dy = Math.abs(y - playerChunkY);
      const dz = Math.abs(z - playerChunkZ);
      if (
        dx > LOAD_RADIUS_XZ + UNLOAD_BUFFER_XZ ||
        dy > LOAD_RADIUS_Y + UNLOAD_BUFFER_Y ||
        dz > LOAD_RADIUS_XZ + UNLOAD_BUFFER_XZ
      ) {
        log("Main", `Unloading chunk: ${key}`);
        chunkMeshes.delete(key);
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

    // --- Rendering ---
    const renderResult = rendererState.renderFrame(
      playerState.getCameraPosition(),
      cameraPitch,
      cameraYaw,
      chunkMeshes,
      debugCameraPosition,
      debugCameraTarget,
      DEBUG_MODE
    );
    lastTotalTriangles = renderResult.totalTriangles;

    const playerChunkX = Math.floor(playerState.position[0] / CHUNK_SIZE_X);
    const playerChunkY = Math.floor(playerState.position[1] / CHUNK_SIZE_Y);
    const playerChunkZ = Math.floor(playerState.position[2] / CHUNK_SIZE_Z);

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
          const chunkPos = {
            x: playerChunkX + xOffset,
            y: playerChunkY + yOffset,
            z: playerChunkZ + zOffset,
          };
          const key = getChunkKey(chunkPos);
          if (!requestedChunkKeys.has(key)) {
            log(
              "Chunk",
              `Requesting chunk: ${key} (Worker ${nextWorkerIndex + 1})`
            );
            requestedChunkKeys.add(key);
            workers[nextWorkerIndex].postMessage({
              type: "requestChunk",
              position: chunkPos,
            });
            nextWorkerIndex = (nextWorkerIndex + 1) % workers.length;
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
Chunk:  (${playerChunkX}, ${playerChunkY}, ${playerChunkZ})
Look:   (${lookDirection[0].toFixed(2)}, ${lookDirection[1].toFixed(
        2
      )}, ${lookDirection[2].toFixed(2)})
Chunks: ${chunkMeshes.size} (${requestedChunkKeys.size} req)
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

    frame(deltaTime);
    physicsStep(deltaTime);
    requestAnimationFrame(frameLoop);
  };

  requestAnimationFrame(frameLoop);

  window.addEventListener("resize", () => {
    rendererState?.resize(window.innerWidth, window.innerHeight);
  });
}

main().catch((err) => {
  log.error("Main", err);
});
