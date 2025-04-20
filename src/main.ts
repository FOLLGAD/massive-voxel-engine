/// <reference types="bun-types" />
/// <reference types="@webgpu/types" />

import { vec3 } from "gl-matrix"; // Keep gl-matrix for look direction vector
import { ENABLE_GREEDY_MESHING } from "./config";
import {
  CHUNK_SIZE_X,
  CHUNK_SIZE_Y,
  CHUNK_SIZE_Z,
  PHYSICS_FPS,
} from "./common/constants";
import { PlayerState, updatePhysics } from "./physics"; // Import physics
import {
  type RendererState,
  initializeRenderer,
  renderFrame,
} from "./renderer"; // Import renderer
import { getChunkKey, type ChunkMesh } from "./chunk";
import log from "./logger";

log("Main", "Main script loaded.");

// --- Global State ---
const chunkMeshes = new Map<string, ChunkMesh>();
const loadedChunkData = new Map<string, Uint8Array>();
const requestedChunkKeys = new Set<string>();
let playerState = new PlayerState();
let rendererState: RendererState | null = null; // Will be initialized later

// --- Camera/Input State ---
let cameraYaw = Math.PI / 4;
let cameraPitch = -Math.PI / 8;
let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;
const MOUSE_SENSITIVITY = 0.005;
const pressedKeys = new Set<string>();

// --- Chunk Loading Config ---
const LOAD_RADIUS_XZ = 4;
const LOAD_RADIUS_Y = 2;
const UNLOAD_BUFFER_XZ = 2;
const UNLOAD_BUFFER_Y = 1;

// --- FPS Calculation State ---
const frameTimes: number[] = [];
const maxFrameSamples = 60;
let lastFrameTime = performance.now();

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

  // --- Input Listeners ---
  canvas.addEventListener("mousedown", (e) => {
    if (e.button === 0) {
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.style.cursor = "grabbing";
    }
  });
  canvas.addEventListener("mouseup", (e) => {
    if (e.button === 0) {
      isDragging = false;
      canvas.style.cursor = "default";
    }
  });
  canvas.addEventListener("mouseleave", () => {
    isDragging = false;
    canvas.style.cursor = "default";
  });
  canvas.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;
    cameraYaw -= deltaX * MOUSE_SENSITIVITY;
    cameraPitch -= deltaY * MOUSE_SENSITIVITY;
    const pitchLimit = Math.PI / 2 - 0.01;
    cameraPitch = Math.max(-pitchLimit, Math.min(pitchLimit, cameraPitch));
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });
  window.addEventListener("keydown", (e) => {
    pressedKeys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    pressedKeys.delete(e.code);
  });

  // --- Initialize Renderer ---
  try {
    rendererState = await initializeRenderer(canvas);
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
    log("Main", `Received message type: ${type}`);

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
        chunkMeshes.set(key, {
          position: position,
          vertexBuffer: vertexBuffer,
          indexBuffer: indexBuffer,
          indexCount: indices.length,
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
      log("Main", `Received empty mesh confirmation for chunk ${key}`);
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

  const physicsStep = (deltaTime: number) => {
    // --- Physics Update ---
    playerState = updatePhysics(
      playerState,
      pressedKeys,
      cameraYaw,
      deltaTime / 1000,
      loadedChunkData
    );
  };

  setInterval(() => physicsStep(1000 / PHYSICS_FPS), 1000 / PHYSICS_FPS); // 30 fps physics

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
  function frame() {
    if (!rendererState) return; // Renderer must be initialized

    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    lastFrameTime = now;

    // --- Rendering ---
    const renderResult = renderFrame(
      rendererState,
      canvas,
      playerState.getCameraPosition(),
      cameraPitch,
      cameraYaw,
      chunkMeshes
    );
    rendererState.depthTexture = renderResult.updatedDepthTexture;
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
              "Main",
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
Tris:   ${lastTotalTriangles.toLocaleString()}
FPS:    ${fps.toFixed(1)}
Mesh:   ${meshingMode}
Gnd:    ${playerState.isGrounded} VelY: ${playerState.velocity[1].toFixed(2)}
        `.trim();
    }

    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);

  window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  });
}

main().catch((err) => {
  log.error("Main", err);
});
