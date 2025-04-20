/// <reference types="bun-types" />
/// <reference types="@webgpu/types" />

import { mat4, vec3 } from "gl-matrix"; // Import gl-matrix
import { ENABLE_GREEDY_MESHING } from "./config";
import { CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z } from "./common/constants";
console.log("Main script loaded.");

// Simple structure to hold mesh data and GPU buffers
interface ChunkMesh {
  position: { x: number; y: number; z: number };
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
}

// Store loaded chunk meshes
const chunkMeshes = new Map<string, ChunkMesh>();
const requestedChunkKeys = new Set<string>(); // Track requested chunks

function getChunkKey(pos: { x: number; y: number; z: number }): string {
  return `${pos.x},${pos.y},${pos.z}`;
}

// Removed inline shader code - assuming it's loaded or still inline below
// Ensure voxelShaderCode is defined somewhere before shaderModule creation
// (Keeping the inline version for now for simplicity)
const voxelShaderCode = `
// Uniforms
struct Uniforms {
    mvpMatrix: mat4x4<f32>,
};
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

// Vertex shader input structure
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) normal: vec3<f32>, // Add normal attribute
};

// Vertex shader output structure (interpolated)
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>, // Pass normal to fragment shader
};

// Vertex Shader
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvpMatrix * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.normal = in.normal; // Pass normal through
    return out;
}

// Fragment Shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    /* // DEBUG: Output normals as color
    let n_len = length(in.normal);
    let surface_normal = select(vec3f(0.0, 0.0, 1.0), normalize(in.normal), n_len > 0.001);
    let normal_color = surface_normal * 0.5 + 0.5;
    return vec4<f32>(normal_color, 1.0);
    */

    // Original lighting code:
    let light_direction = normalize(vec3<f32>(0.8, 0.6, 0.2)); // More directional light
    let ambient_light = 0.3;

    // Ensure normal is normalized after interpolation, handle potential zero vectors
    let n_len = length(in.normal);
    let surface_normal = select(
        vec3f(0.0, 0.0, 1.0), // Default normal if length is near zero
        normalize(in.normal), // Use normalized normal otherwise
        n_len > 0.001 // Condition to check
    );

    // Calculate diffuse lighting (dot product, clamped)
    let diffuse_intensity = max(dot(surface_normal, light_direction), 0.0);

    // Combine ambient and diffuse
    let brightness = ambient_light + (1.0 - ambient_light) * diffuse_intensity;

    // Apply brightness to the color
    let final_color = in.color * brightness;

    return vec4<f32>(final_color, 1.0);
}
`;

// --- Camera State ---
let cameraYaw = Math.PI / 4; // Initial horizontal rotation
let cameraPitch = -Math.PI / 8; // Initial vertical rotation (look slightly down)
const cameraUp = vec3.fromValues(0, 1, 0); // World up direction
const cameraPosition = vec3.fromValues(
  CHUNK_SIZE_X / 2,
  CHUNK_SIZE_Y + 5,
  CHUNK_SIZE_Z / 2
); // Start above center of first chunk

let isDragging = false;
let lastMouseX = 0;
let lastMouseY = 0;
const MOUSE_SENSITIVITY = 0.005;
const MOVEMENT_SPEED = 0.05; // Units per second (Reduced by 100x)
const LOAD_RADIUS_XZ = 4; // Load chunks in an N*2+1 x N*2+1 area around the player horizontally
const LOAD_RADIUS_Y = 2; // Load chunks N levels above and below the player vertically
const UNLOAD_BUFFER_XZ = 2; // Unload chunks XZ distance > LOAD_RADIUS_XZ + this buffer
const UNLOAD_BUFFER_Y = 1; // Unload chunks Y distance > LOAD_RADIUS_Y + this buffer

// --- FPS Calculation State ---
const frameTimes: number[] = [];
const maxFrameSamples = 60; // Average FPS over this many frames
let lastFrameTime = performance.now();

// --- Helper Functions (using gl-matrix) ---
function updateViewMatrix(
  viewMatrix: mat4,
  eye: vec3,
  pitch: number,
  yaw: number
) {
  const direction = vec3.create();
  direction[0] = Math.cos(pitch) * Math.sin(yaw);
  direction[1] = Math.sin(pitch);
  direction[2] = Math.cos(pitch) * Math.cos(yaw);
  vec3.normalize(direction, direction);

  const center = vec3.create();
  vec3.add(center, eye, direction); // Calculate look-at point

  mat4.lookAt(viewMatrix, eye, center, cameraUp);
}

async function main() {
  const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
  if (!canvas) {
    console.error("Canvas element not found!");
    return;
  }
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  // Prevent default context menu on right-click (optional)
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  // --- Mouse Event Listeners ---
  canvas.addEventListener("mousedown", (e) => {
    // Check for left mouse button (button === 0)
    if (e.button === 0) {
      isDragging = true;
      lastMouseX = e.clientX;
      lastMouseY = e.clientY;
      canvas.style.cursor = "grabbing"; // Change cursor style
    }
  });

  canvas.addEventListener("mouseup", (e) => {
    if (e.button === 0) {
      isDragging = false;
      canvas.style.cursor = "default"; // Reset cursor style
    }
  });

  canvas.addEventListener("mouseleave", () => {
    // Stop dragging if mouse leaves canvas
    isDragging = false;
    canvas.style.cursor = "default";
  });

  canvas.addEventListener("mousemove", (e) => {
    if (!isDragging) return;

    const deltaX = e.clientX - lastMouseX;
    const deltaY = e.clientY - lastMouseY;

    cameraYaw -= deltaX * MOUSE_SENSITIVITY;
    cameraPitch -= deltaY * MOUSE_SENSITIVITY;

    // Clamp pitch to avoid flipping
    const pitchLimit = Math.PI / 2 - 0.01; // Slightly less than 90 degrees
    cameraPitch = Math.max(-pitchLimit, Math.min(pitchLimit, cameraPitch));

    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });

  // Keyboard Listeners
  const pressedKeys = new Set<string>(); // Keep track of pressed keys
  window.addEventListener("keydown", (e) => {
    pressedKeys.add(e.code);
  });
  window.addEventListener("keyup", (e) => {
    pressedKeys.delete(e.code);
  });

  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }

  const device = await adapter.requestDevice();

  const context = canvas.getContext("webgpu") as GPUCanvasContext | null;
  if (!context) {
    throw new Error("Could not get WebGPU context from canvas.");
  }

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: presentationFormat,
    alphaMode: "opaque",
  });

  console.log("WebGPU Initialized");

  // Initialize worker pool
  const numWorkers = navigator.hardwareConcurrency || 4; // Use hardware concurrency or default to 4
  console.log(`Initializing ${numWorkers} workers...`);
  const workers: Worker[] = [];
  for (let i = 0; i < numWorkers; i++) {
    console.log(`Initializing worker ${i + 1}/${numWorkers}...`);
    workers.push(new Worker("./worker.js", { type: "module" }));
  }
  let nextWorkerIndex = 0;

  console.log(`${numWorkers} Workers initialized`);

  const workerMessageHandler = (event: MessageEvent) => {
    console.log(
      `[Main] Received message type: ${event.data.type} from a worker`
    ); // Log received message type

    if (event.data.type === "chunkMeshAvailable") {
      const {
        position,
        vertices: verticesBuffer,
        indices: indicesBuffer,
      } = event.data;
      const vertices = new Float32Array(verticesBuffer);
      const indices = new Uint32Array(indicesBuffer);

      // Add try...catch around buffer creation/writing
      try {
        // Create GPU Buffers (label might need adjustment if getChunkKey returns complex string)
        const vertexBuffer = device.createBuffer({
          label: `chunk-${position.x}-${position.y}-${position.z}-vertex`, // Simpler label
          size: vertices.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(vertexBuffer, 0, vertices);

        const indexBuffer = device.createBuffer({
          label: `chunk-${position.x}-${position.y}-${position.z}-index`,
          size: indices.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(indexBuffer, 0, indices);

        // Store the mesh data and buffers
        chunkMeshes.set(getChunkKey(position), {
          position: position,
          vertexBuffer: vertexBuffer,
          indexBuffer: indexBuffer,
          indexCount: indices.length,
        });
        console.log(`[Main] Stored mesh for ${getChunkKey(position)}`); // Confirm storage
        // Ensure key is marked as requested even if mesh generation was delayed
        requestedChunkKeys.add(getChunkKey(position));
      } catch (error) {
        console.error("[Main] Error creating/writing GPU buffers:", error); // Log buffer errors
      }
    } else if (event.data.type === "chunkMeshEmpty") {
      console.log(
        `[Main] Received empty mesh confirmation for chunk ${getChunkKey(
          event.data.position
        )}`
      );
      // Mark as requested so we don't ask again
      requestedChunkKeys.add(getChunkKey(event.data.position));
    } else if (event.data.type === "chunkNotAvailable") {
      console.log(
        `[Main] Chunk not available: ${JSON.stringify(event.data.position)}`
      );
    } else {
      console.warn("[Main] Unknown message type from worker:", event.data.type);
    }
  };

  // Attach the handler to each worker
  for (const worker of workers) {
    worker.onmessage = workerMessageHandler;
  }

  // --- Rendering Setup ---

  const shaderModule = device.createShaderModule({
    label: "Voxel Shader Module",
    code: voxelShaderCode,
  });

  // Vertex Buffer Layout - UPDATED for normals
  const vertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 9 * Float32Array.BYTES_PER_ELEMENT, // 3 pos + 3 color + 3 normal
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" }, // Position
      {
        shaderLocation: 1,
        offset: 3 * Float32Array.BYTES_PER_ELEMENT,
        format: "float32x3",
      }, // Color
      {
        shaderLocation: 2,
        offset: 6 * Float32Array.BYTES_PER_ELEMENT,
        format: "float32x3",
      }, // Normal
    ],
  };

  // MVP Matrix Uniform Buffer (using gl-matrix type)
  const mvpMatrix = mat4.create();
  const uniformBufferSize = 16 * Float32Array.BYTES_PER_ELEMENT;
  const uniformBuffer = device.createBuffer({
    label: "Uniform Buffer (MVP Matrix)",
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  // Create Bind Group Layout (describes bindings for a pipeline)
  const bindGroupLayout = device.createBindGroupLayout({
    label: "Voxel Bind Group Layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });

  // Create Bind Group (connects resources to bindings)
  const bindGroup = device.createBindGroup({
    label: "Voxel Bind Group",
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: { buffer: uniformBuffer },
      },
    ],
  });

  // Create Pipeline Layout
  const pipelineLayout = device.createPipelineLayout({
    label: "Voxel Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout], // Use the layout created above
  });

  // Depth Texture
  let depthTexture: GPUTexture;
  function configureDepthTexture() {
    if (depthTexture) {
      depthTexture.destroy();
    }
    depthTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: "depth24plus", // Standard depth format
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }
  configureDepthTexture(); // Initial configuration

  // Render Pipeline (uses the updated vertexBufferLayout)
  const pipeline = device.createRenderPipeline({
    label: "Voxel Render Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: shaderModule,
      entryPoint: "vs_main",
      buffers: [vertexBufferLayout], // Pass the updated layout
    },
    fragment: {
      module: shaderModule,
      entryPoint: "fs_main",
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "back", // Assuming culling should be back on
    },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });

  // --- Matrices (using gl-matrix) ---
  const viewMatrix = mat4.create();
  const projectionMatrix = mat4.create();

  const debugInfoElement = document.getElementById(
    "debug-info"
  ) as HTMLDivElement;
  if (!debugInfoElement) {
    console.error("Debug info element not found!");
    // Continue without debug info if not found
  }

  // --- Game Loop ---
  function frame() {
    if (!context) {
      console.error("Context is null");
      return;
    }

    const now = performance.now();
    const deltaTime = now - lastFrameTime;
    lastFrameTime = now;

    // Update FPS - Calculate average over last N frames
    frameTimes.push(deltaTime);
    if (frameTimes.length > maxFrameSamples) {
      frameTimes.shift(); // Remove oldest frame time
    }
    let averageDeltaTime = 0;
    if (frameTimes.length > 0) {
      averageDeltaTime =
        frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
    }
    const fps = averageDeltaTime > 0 ? 1000 / averageDeltaTime : 0;

    // Configure depth texture if needed
    if (
      !depthTexture ||
      canvas.width !== depthTexture.width ||
      canvas.height !== depthTexture.height
    ) {
      configureDepthTexture();
    }

    // Update View Matrix based on camera rotation
    updateViewMatrix(viewMatrix, cameraPosition, cameraPitch, cameraYaw);

    // Update Projection Matrix
    const aspect = canvas.width / canvas.height;
    mat4.perspective(projectionMatrix, Math.PI / 4, aspect, 0.1, 1000.0);

    // Calculate MVP Matrix (Model is identity for now)
    mat4.multiply(mvpMatrix, projectionMatrix, viewMatrix);
    // mat4.multiply(mvpMatrix, mvpMatrix, modelMatrix); // If needed

    // Update Uniform Buffer
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      mvpMatrix as Float32Array // Cast gl-matrix mat4 to Float32Array for writeBuffer
    );

    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    // Set pipeline and bind group
    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);

    // Calculate total triangles rendered
    let totalTriangles = 0;
    for (const mesh of chunkMeshes.values()) {
      passEncoder.setVertexBuffer(0, mesh.vertexBuffer);
      passEncoder.setIndexBuffer(mesh.indexBuffer, "uint32");
      passEncoder.drawIndexed(mesh.indexCount);
      totalTriangles += mesh.indexCount / 3; // 3 indices per triangle
    }

    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);

    // --- Chunk Unloading ---
    const camChunkXUnload = Math.floor(cameraPosition[0] / CHUNK_SIZE_X);
    const camChunkYUnload = Math.floor(cameraPosition[1] / CHUNK_SIZE_Y);
    const camChunkZUnload = Math.floor(cameraPosition[2] / CHUNK_SIZE_Z);

    for (const [key, chunkMesh] of chunkMeshes.entries()) {
      const { x, y, z } = chunkMesh.position; // Chunk's integer position

      const dx = Math.abs(x - camChunkXUnload);
      const dy = Math.abs(y - camChunkYUnload);
      const dz = Math.abs(z - camChunkZUnload);

      // Check if the chunk is outside the rendering distance plus the buffer
      if (
        dx > LOAD_RADIUS_XZ + UNLOAD_BUFFER_XZ ||
        dy > LOAD_RADIUS_Y + UNLOAD_BUFFER_Y ||
        dz > LOAD_RADIUS_XZ + UNLOAD_BUFFER_XZ
      ) {
        console.log(`[Main] Unloading chunk: ${key}`);
        chunkMesh.vertexBuffer.destroy();
        chunkMesh.indexBuffer.destroy();
        chunkMeshes.delete(key);
        requestedChunkKeys.delete(key); // Also remove from requested set if it was unloaded before it finished loading
      }
    }

    // --- Dynamic Chunk Loading (uses separate cam chunk coords for clarity) ---
    const camChunkXLoad = Math.floor(cameraPosition[0] / CHUNK_SIZE_X);
    const camChunkYLoad = Math.floor(cameraPosition[1] / CHUNK_SIZE_Y);
    const camChunkZLoad = Math.floor(cameraPosition[2] / CHUNK_SIZE_Z);

    // --- Update Debug Info ---
    if (debugInfoElement) {
      const avgDelta =
        frameTimes.length > 0
          ? frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length
          : 0;
      const fps = avgDelta > 0 ? 1000 / avgDelta : 0;
      frameTimes.push(avgDelta);
      if (frameTimes.length > maxFrameSamples) frameTimes.shift();

      // Calculate direction vector for debug display
      const lookDirection = vec3.create();
      lookDirection[0] = Math.cos(cameraPitch) * Math.sin(cameraYaw);
      lookDirection[1] = Math.sin(cameraPitch);
      lookDirection[2] = Math.cos(cameraPitch) * Math.cos(cameraYaw);
      vec3.normalize(lookDirection, lookDirection);

      const meshingMode = ENABLE_GREEDY_MESHING ? "Greedy" : "Naive";
      debugInfoElement.textContent = `
Pos:    (${cameraPosition[0].toFixed(1)}, ${cameraPosition[1].toFixed(
        1
      )}, ${cameraPosition[2].toFixed(1)})
Chunk:  (${camChunkXLoad}, ${camChunkYLoad}, ${camChunkZLoad}) // Use Load coords for debug
Look:   (${lookDirection[0].toFixed(2)}, ${lookDirection[1].toFixed(
        2
      )}, ${lookDirection[2].toFixed(2)})
Chunks: ${chunkMeshes.size} (${requestedChunkKeys.size} req)
Tris:   ${totalTriangles.toLocaleString()}
FPS:    ${fps.toFixed(1)}
Mesh:   ${meshingMode}
        `.trim();
    }

    // --- Handle Input / Camera Movement ---
    const forward = vec3.create();
    forward[0] = Math.sin(cameraYaw);
    forward[2] = Math.cos(cameraYaw);
    vec3.normalize(forward, forward); // Horizontal forward vector

    const right = vec3.create();
    vec3.cross(right, forward, cameraUp);
    vec3.normalize(right, right); // Horizontal right vector

    const speed = MOVEMENT_SPEED * averageDeltaTime;
    if (pressedKeys.has("KeyW")) {
      vec3.scaleAndAdd(cameraPosition, cameraPosition, forward, speed);
    }
    if (pressedKeys.has("KeyS")) {
      vec3.scaleAndAdd(cameraPosition, cameraPosition, forward, -speed);
    }
    if (pressedKeys.has("KeyA")) {
      vec3.scaleAndAdd(cameraPosition, cameraPosition, right, -speed);
    }
    if (pressedKeys.has("KeyD")) {
      vec3.scaleAndAdd(cameraPosition, cameraPosition, right, speed);
    }
    if (pressedKeys.has("Space")) {
      cameraPosition[1] += speed;
    } // Move up along world Y
    if (pressedKeys.has("ShiftLeft")) {
      cameraPosition[1] -= speed;
    } // Move down along world Y

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
            x: camChunkXLoad + xOffset,
            y: camChunkYLoad + yOffset,
            z: camChunkZLoad + zOffset,
          };
          const key = getChunkKey(chunkPos);

          if (!requestedChunkKeys.has(key)) {
            console.log(
              `[Main] Requesting chunk: ${key} (Worker ${nextWorkerIndex + 1})`
            );
            requestedChunkKeys.add(key);
            // Send to the next worker in the pool (round-robin)
            workers[nextWorkerIndex].postMessage({
              type: "requestChunk",
              position: chunkPos,
            });
            nextWorkerIndex = (nextWorkerIndex + 1) % workers.length;
          }
        }
      }
    }

    requestAnimationFrame(frame);
  } // End frame loop

  requestAnimationFrame(frame);

  // Handle window resizing
  window.addEventListener("resize", () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    // Depth texture is recreated in the frame loop if size mismatch
  });
}

main().catch((err) => {
  console.error(err);
});
