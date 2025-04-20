import { mat4, vec3, vec4 } from "gl-matrix";
import type { ChunkMesh } from "./chunk";
import { PLAYER_HEIGHT, PLAYER_WIDTH } from "./physics";


// --- Constants ---
const FRUSTUM_CULLING_EPSILON = 1e-5;
const DEBUG_COLOR_CULLED = [1, 0, 0]; // Red
const DEBUG_COLOR_DRAWN = [0, 1, 0]; // Green
const DEBUG_COLOR_FRUSTUM = [0.8, 0.8, 0]; // Yellow
const DEBUG_COLOR_PLAYER = [0, 0.8, 0.8]; // Cyan
const INITIAL_DEBUG_LINE_BUFFER_SIZE = 1024 * 6 * 4 * 10; // ~1k lines

// --- Renderer State ---
export interface RendererState {
  device: GPUDevice;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  voxelPipeline: GPURenderPipeline;
  linePipeline: GPURenderPipeline; // For drawing debug lines
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  depthTexture: GPUTexture;
  debugLineBuffer: GPUBuffer; // Dynamic buffer for line vertices
  debugLineBufferSize: number; // Current size of the buffer
  // Reusable matrices
  viewMatrix: mat4;
  projectionMatrix: mat4;
  vpMatrix: mat4; // View * Projection
  // Reusable matrices for Debug Camera
  viewMatrixDebug: mat4;
  projectionMatrixDebug: mat4;
  vpMatrixDebug: mat4; // Debug View * Debug Projection
  // Debug Info
  debugInfo: {
    totalChunks: number;
    drawnChunks: number;
    totalTriangles: number;
    culledChunks: number;
  };
}

// @ts-ignore
import voxelShaderCode from "./shaders/voxel.wsgl" with { type: "text" };
// @ts-ignore
import lineShaderCode from "./shaders/line.wsgl" with { type: "text" };
// @ts-ignore
import cullChunksShader from "./shaders/cullChunks.wsgl" with { type: "text" };

// --- Frustum Culling Helpers ---

/**
 * Represents a plane equation: Ax + By + Cz + D = 0
 * The normal vector is (A, B, C).
 */
type Plane = vec4; // [A, B, C, D]

/**
 * Extracts the 6 planes of the viewing frustum from a combined view-projection matrix.
 * Plane normals point inwards.
 * @param vpMatrix The combined view-projection matrix.
 * @returns An array of 6 planes [left, right, bottom, top, near, far].
 */
function extractFrustumPlanes(mat: mat4): Plane[] {
  const planes: Plane[] = [
    vec4.create(), // Left
    vec4.create(), // Right
    vec4.create(), // Bottom
    vec4.create(), // Top
    vec4.create(), // Near
    vec4.create(), // Far
  ];
  const m = mat; // Use the input parameter
  const get = (i: number, j: number) => m[i * 4 + j];

  for (let i = 4; i--; ) {
    planes[0][i] = get(i, 3) + get(i, 0);
  }
  for (let i = 4; i--; ) {
    planes[1][i] = get(i, 3) - get(i, 0);
  }
  for (let i = 4; i--; ) {
    planes[2][i] = get(i, 3) + get(i, 1);
  }
  for (let i = 4; i--; ) {
    planes[3][i] = get(i, 3) - get(i, 1);
  }
  for (let i = 4; i--; ) {
    planes[4][i] = get(i, 3) + get(i, 2);
  }
  for (let i = 4; i--; ) {
    planes[5][i] = get(i, 3) - get(i, 2);
  }

  // Normalize the plane equations
  for (const plane of planes) {
    const invLength = 1.0 / vec3.length([plane[0], plane[1], plane[2]]);
    vec4.scale(plane, plane, invLength);
  }

  return planes;
}

/**
 * Checks if an AABB intersects with the view frustum.
 * Uses the "positive/negative vertex" optimization.
 * Assumes plane normals point inwards.
 * @param planes The 6 frustum planes.
 * @param aabb The Axis-Aligned Bounding Box { min: vec3, max: vec3 }.
 * @returns true if the AABB intersects the frustum, false otherwise.
 */
function intersectFrustumAABB(
  planes: Plane[],
  aabb: { min: vec3; max: vec3 }
): boolean {
  const { min, max } = aabb;

  for (let i = 0; i < 6; i++) {
    const plane = planes[i];
    const normalX = plane[0];
    const normalY = plane[1];
    const normalZ = plane[2];
    const planeD = plane[3];

    // Find the positive vertex (P-vertex) - furthest along the normal
    const positiveVertex: vec3 = [
      normalX > 0 ? max[0] : min[0],
      normalY > 0 ? max[1] : min[1],
      normalZ > 0 ? max[2] : min[2],
    ];

    // Check if the positive vertex is outside the plane (behind it)
    // If distance(P) < 0, the entire box is outside this plane
    const distance =
      normalX * positiveVertex[0] +
      normalY * positiveVertex[1] +
      normalZ * positiveVertex[2] +
      planeD;

    if (distance < -FRUSTUM_CULLING_EPSILON) {
      return false;
    }
  }

  // If the AABB wasn't fully outside any single plane, it must intersect or be inside
  return true;
}

// --- Add Helper to Get Frustum Corners ---
/**
 * Calculates the 8 corners of the view frustum in world space.
 * @param invVpMatrix The inverse of the combined view-projection matrix.
 * @returns An array of 8 vec3 points representing the frustum corners.
 *          Order: near_bottom_left, near_top_left, near_top_right, near_bottom_right,
 *                 far_bottom_left,  far_top_left,  far_top_right,  far_bottom_right
 */
function getFrustumCornersWorldSpace(invVpMatrix: mat4): vec3[] {
  const ndcCorners: vec4[] = [
    // Near face
    vec4.fromValues(-1, -1, -1, 1), // near_bottom_left (in NDC before perspective divide)
    vec4.fromValues(-1, 1, -1, 1), // near_top_left
    vec4.fromValues(1, 1, -1, 1), // near_top_right
    vec4.fromValues(1, -1, -1, 1), // near_bottom_right
    // Far face
    vec4.fromValues(-1, -1, 1, 1), // far_bottom_left
    vec4.fromValues(-1, 1, 1, 1), // far_top_left
    vec4.fromValues(1, 1, 1, 1), // far_top_right
    vec4.fromValues(1, -1, 1, 1), // far_bottom_right
  ];

  const worldCorners: vec3[] = [];
  for (const ndc of ndcCorners) {
    const worldVec4 = vec4.create();
    vec4.transformMat4(worldVec4, ndc, invVpMatrix);
    // Perspective divide
    if (worldVec4[3] !== 0) {
      vec4.scale(worldVec4, worldVec4, 1.0 / worldVec4[3]);
    }
    worldCorners.push(
      vec3.fromValues(worldVec4[0], worldVec4[1], worldVec4[2])
    );
  }
  return worldCorners;
}

// --- Helper Functions ---

function updateViewMatrix(
  viewMatrix: mat4,
  eye: vec3,
  pitch: number,
  yaw: number,
  up: vec3 = vec3.fromValues(0, 1, 0)
) {
  const direction = vec3.create();
  direction[0] = Math.cos(pitch) * Math.sin(yaw);
  direction[1] = Math.sin(pitch);
  direction[2] = Math.cos(pitch) * Math.cos(yaw);
  vec3.normalize(direction, direction);

  const center = vec3.create();
  vec3.add(center, eye, direction);

  mat4.lookAt(viewMatrix, eye, center, up);
}

function configureDepthTexture(
  device: GPUDevice,
  canvas: HTMLCanvasElement,
  currentDepthTexture: GPUTexture | null
): GPUTexture {
  if (
    currentDepthTexture &&
    currentDepthTexture.width === canvas.width &&
    currentDepthTexture.height === canvas.height
  ) {
    return currentDepthTexture;
  }
  if (currentDepthTexture) {
    currentDepthTexture.destroy();
  }
  return device.createTexture({
    label: "Depth Texture",
    size: [canvas.width, canvas.height],
    format: "depth24plus",
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

// --- Debug Drawing Helpers ---

/** Calculates the player's AABB based on camera position (eye level) */
function getPlayerAABB(cameraPos: vec3): { min: vec3; max: vec3 } {
  const halfWidth = PLAYER_WIDTH / 2;
  const minY = cameraPos[1] - PLAYER_HEIGHT;
  const maxY = cameraPos[1];
  return {
    min: vec3.fromValues(
      cameraPos[0] - halfWidth,
      minY,
      cameraPos[2] - halfWidth
    ),
    max: vec3.fromValues(
      cameraPos[0] + halfWidth,
      maxY,
      cameraPos[2] + halfWidth
    ),
  };
}

/** Adds line vertices (pos[3] + color[3]) for the 12 edges of an AABB */
function addAABBLineVertices(
  vertices: number[],
  aabb: { min: vec3; max: vec3 },
  color: number[]
) {
  const { min, max } = aabb;
  // Define the 8 corners
  const corners = [
    vec3.fromValues(min[0], min[1], min[2]), // 0: --- Near Bottom Left
    vec3.fromValues(max[0], min[1], min[2]), // 1: +-- Near Bottom Right
    vec3.fromValues(max[0], max[1], min[2]), // 2: ++- Near Top Right
    vec3.fromValues(min[0], max[1], min[2]), // 3: -+- Near Top Left
    vec3.fromValues(min[0], min[1], max[2]), // 4: --+ Far Bottom Left
    vec3.fromValues(max[0], min[1], max[2]), // 5: +-+ Far Bottom Right
    vec3.fromValues(max[0], max[1], max[2]), // 6: +++ Far Top Right
    vec3.fromValues(min[0], max[1], max[2]), // 7: -++ Far Top Left
  ];

  // Define the 12 lines by connecting corner indices
  const lines = [
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    0, // Bottom face
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    4, // Top face
    0,
    4,
    1,
    5,
    2,
    6,
    3,
    7, // Connecting sides
  ];

  for (let i = 0; i < lines.length; i += 2) {
    const c1 = corners[lines[i]];
    const c2 = corners[lines[i + 1]];
    // Add vertex 1 (pos + color)
    vertices.push(c1[0], c1[1], c1[2], color[0], color[1], color[2]);
    // Add vertex 2 (pos + color)
    vertices.push(c2[0], c2[1], c2[2], color[0], color[1], color[2]);
  }
}

/** Adds line vertices for the 12 edges of a frustum defined by its 8 corners */
function addFrustumLineVertices(
  vertices: number[],
  corners: vec3[],
  color: number[]
) {
  if (corners.length !== 8) return; // Need exactly 8 corners

  // Define the 12 lines by connecting corner indices (same as AABB)
  // Corner order expected: NBL, NTL, NTR, NBR, FBL, FTL, FTR, FBR
  const lines = [
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    0, // Near face
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    4, // Far face
    0,
    4,
    1,
    5,
    2,
    6,
    3,
    7, // Connecting sides
  ];

  for (let i = 0; i < lines.length; i += 2) {
    const c1 = corners[lines[i]];
    const c2 = corners[lines[i + 1]];
    // Add vertex 1 (pos + color)
    vertices.push(c1[0], c1[1], c1[2], color[0], color[1], color[2]);
    // Add vertex 2 (pos + color)
    vertices.push(c2[0], c2[1], c2[2], color[0], color[1], color[2]);
  }
}

/** Generates the Float32Array vertex data for all debug lines */
function generateDebugLineVertices(
  rendererState: RendererState,
  chunkMeshes: Map<string, ChunkMesh>,
  frustumPlanes: Plane[],
  worldFrustumCorners: vec3[],
  cameraPosition: vec3
): Float32Array | null {
  const lineVertices: number[] = [];

  // Generate lines for ALL culled/drawn chunks (based on FP camera)
  for (const mesh of chunkMeshes.values()) {
    const intersects = intersectFrustumAABB(frustumPlanes, mesh.aabb);
    addAABBLineVertices(
      lineVertices,
      mesh.aabb,
      intersects ? DEBUG_COLOR_DRAWN : DEBUG_COLOR_CULLED
    );
  }

  // Add first-person frustum lines
  addFrustumLineVertices(
    lineVertices,
    worldFrustumCorners,
    DEBUG_COLOR_FRUSTUM
  );

  // Add player hitbox lines
  const playerAABB = getPlayerAABB(cameraPosition);
  addAABBLineVertices(lineVertices, playerAABB, DEBUG_COLOR_PLAYER);

  // Prepare buffer data
  const lineData = new Float32Array(lineVertices);
  if (lineData.byteLength > rendererState.debugLineBufferSize) {
    console.warn("Renderer", "Debug line buffer too small, resizing needed!");
    return null; // Don't draw if buffer is too small
  }
  return lineData.length > 0 ? lineData : null;
}

// --- Pipeline Creation ---

function createVoxelPipeline(
  device: GPUDevice,
  presentationFormat: GPUTextureFormat,
  pipelineLayout: GPUPipelineLayout
): GPURenderPipeline {
  const voxelShaderModule = device.createShaderModule({
    code: voxelShaderCode,
  });
  const voxelVertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 9 * Float32Array.BYTES_PER_ELEMENT,
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" }, // Position
      { shaderLocation: 1, offset: 3 * 4, format: "float32x3" }, // Color
      { shaderLocation: 2, offset: 6 * 4, format: "float32x3" }, // Normal
    ],
  };
  return device.createRenderPipeline({
    label: "Voxel Render Pipeline",
    layout: pipelineLayout,
    vertex: {
      module: voxelShaderModule,
      entryPoint: "vs_main",
      buffers: [voxelVertexBufferLayout],
    },
    fragment: {
      module: voxelShaderModule,
      entryPoint: "fs_main",
      targets: [{ format: presentationFormat }],
    },
    primitive: { topology: "triangle-list", cullMode: "back" },
    depthStencil: {
      depthWriteEnabled: true,
      depthCompare: "less",
      format: "depth24plus",
    },
  });
}

function createLinePipeline(
  device: GPUDevice,
  presentationFormat: GPUTextureFormat,
  pipelineLayout: GPUPipelineLayout
): GPURenderPipeline {
  const lineShaderModule = device.createShaderModule({
    code: lineShaderCode,
  });

  const lineVertexBufferLayout: GPUVertexBufferLayout = {
    arrayStride: 6 * Float32Array.BYTES_PER_ELEMENT, // 3 pos + 3 color
    attributes: [
      { shaderLocation: 0, offset: 0, format: "float32x3" }, // Position
      {
        shaderLocation: 1,
        offset: 3 * Float32Array.BYTES_PER_ELEMENT,
        format: "float32x3",
      }, // Color
    ],
  };

  return device.createRenderPipeline({
    label: "Line Render Pipeline",
    layout: pipelineLayout, // Assumes same bind group 0 for MVP matrix
    vertex: {
      module: lineShaderModule,
      entryPoint: "vs_main",
      buffers: [lineVertexBufferLayout],
    },
    fragment: {
      module: lineShaderModule,
      entryPoint: "fs_main",
      targets: [{ format: presentationFormat }],
    },
    primitive: {
      topology: "line-list", // Use line-list topology
    },
    depthStencil: {
      depthWriteEnabled: true, // Lines should respect depth
      depthCompare: "less", // Restore original depth test
      format: "depth24plus",
    },
  });
}

// --- Initialization ---
export async function initializeRenderer(
  canvas: HTMLCanvasElement
): Promise<RendererState> {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
  }
  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Could not get WebGPU context from canvas.");
  }
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device: device,
    format: presentationFormat,
    alphaMode: "opaque",
  });

  // --- Uniform Buffer, Bind Group, Layout ---
  const uniformBufferSize = 16 * Float32Array.BYTES_PER_ELEMENT; // MVP Matrix
  const uniformBuffer = device.createBuffer({
    label: "Uniform Buffer (MVP Matrix)",
    size: uniformBufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    label: "Shared Pipeline Layout",
    bindGroupLayouts: [bindGroupLayout],
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });

  // --- Pipelines ---
  const voxelPipeline = createVoxelPipeline(
    device,
    presentationFormat,
    pipelineLayout
  );
  const linePipeline = createLinePipeline(
    device,
    presentationFormat,
    pipelineLayout
  );

  // --- Debug Line Buffer ---
  const debugLineBuffer = device.createBuffer({
    label: "Debug Line Buffer",
    size: INITIAL_DEBUG_LINE_BUFFER_SIZE,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  });

  // --- Depth Texture & Matrices ---
  const depthTexture = configureDepthTexture(device, canvas, null);
  const viewMatrix = mat4.create();
  const projectionMatrix = mat4.create();
  const vpMatrix = mat4.create();

  // Debug Camera Matrices
  const viewMatrixDebug = mat4.create();
  const projectionMatrixDebug = mat4.create();
  const vpMatrixDebug = mat4.create();

  // --- Initial State ---
  return {
    device,
    context,
    presentationFormat,
    uniformBuffer,
    bindGroup,
    voxelPipeline,
    linePipeline,
    depthTexture,
    debugLineBuffer,
    debugLineBufferSize: INITIAL_DEBUG_LINE_BUFFER_SIZE,
    viewMatrix,
    projectionMatrix,
    vpMatrix,
    debugInfo: {
      totalChunks: 0,
      drawnChunks: 0,
      totalTriangles: 0,
      culledChunks: 0,
    },
    // Initialize Debug Camera Matrices
    viewMatrixDebug,
    projectionMatrixDebug,
    vpMatrixDebug,
  };
}

// --- Drawing Functions ---

/** Draws the main voxel scene, performing culling */
function drawVoxelScene(
  passEncoder: GPURenderPassEncoder,
  rendererState: RendererState,
  chunkMeshes: Map<string, ChunkMesh>,
  frustumPlanes: Plane[]
): { drawnChunks: number; culledChunks: number; totalTriangles: number } {
  passEncoder.setPipeline(rendererState.voxelPipeline);
  passEncoder.setBindGroup(0, rendererState.bindGroup); // Uses FP VP matrix (already in buffer)

  let totalTriangles = 0;
  let drawnChunks = 0;
  let culledChunks = 0;

  for (const mesh of chunkMeshes.values()) {
    if (intersectFrustumAABB(frustumPlanes, mesh.aabb)) {
      passEncoder.setVertexBuffer(0, mesh.vertexBuffer);
      passEncoder.setIndexBuffer(mesh.indexBuffer, "uint32");
      passEncoder.drawIndexed(mesh.indexCount);
      totalTriangles += mesh.indexCount / 3;
      drawnChunks++;
    } else {
      culledChunks++;
    }
  }
  return { drawnChunks, culledChunks, totalTriangles };
}

/** Draws the debug lines using the provided VP matrix */
function drawDebugLines(
  passEncoder: GPURenderPassEncoder,
  rendererState: RendererState,
  lineData: Float32Array
) {
  // Write the debug view matrix to the uniform buffer
  rendererState.device.queue.writeBuffer(
    rendererState.uniformBuffer,
    0,
    rendererState.vpMatrixDebug as Float32Array
  );
  // Upload the line data to the buffer
  rendererState.device.queue.writeBuffer(
    rendererState.debugLineBuffer,
    0,
    lineData
  );

  // Set pipeline, bind group (which now uses the debug matrix), and buffer
  passEncoder.setPipeline(rendererState.linePipeline);
  passEncoder.setBindGroup(0, rendererState.bindGroup);
  passEncoder.setVertexBuffer(0, rendererState.debugLineBuffer);
  passEncoder.draw(lineData.length / 6);
}

// --- Render Frame ---
export function renderFrame(
  rendererState: RendererState,
  canvas: HTMLCanvasElement,
  cameraPosition: vec3,
  cameraPitch: number,
  cameraYaw: number,
  chunkMeshes: Map<string, ChunkMesh>,
  debugCameraPosition: vec3,
  debugCameraTarget: vec3,
  enableDebugView = true
): {
  updatedDepthTexture: GPUTexture;
  totalTriangles: number;
  drawnChunks: number;
} {
  // Simplified return
  const {
    device,
    context,
    viewMatrix,
    projectionMatrix,
    vpMatrix,
    uniformBuffer,
    viewMatrixDebug,
    projectionMatrixDebug,
    vpMatrixDebug,
    depthTexture: oldDepthTexture,
  } = rendererState;

  const depthTexture = configureDepthTexture(device, canvas, oldDepthTexture);
  const aspect = canvas.width / canvas.height;

  // Calculate main camera matrices
  // First Person
  updateViewMatrix(viewMatrix, cameraPosition, cameraPitch, cameraYaw);
  mat4.perspective(projectionMatrix, Math.PI / 4, aspect, 0.1, 1000.0);
  mat4.multiply(vpMatrix, projectionMatrix, viewMatrix);

  // Calculate inverse main VP for frustum corners
  const invVpMatrix = mat4.create();
  mat4.invert(invVpMatrix, vpMatrix);
  let worldFrustumCorners: vec3[] = [];

  // Calculate debug camera matrices
  if (enableDebugView) {
    // Calculate debug camera view using the provided parameters
    mat4.lookAt(
      // Use the new parameters
      viewMatrixDebug,
      debugCameraPosition,
      debugCameraTarget,
      vec3.fromValues(0, 1, 0)
    );

    // Use the same projection or define a separate one
    mat4.copy(projectionMatrixDebug, projectionMatrix); // Or create a different one if needed
    // Calculate the combined VP matrix for the debug camera
    mat4.multiply(vpMatrixDebug, projectionMatrixDebug, viewMatrixDebug);

    // Calculate main camera frustum corners for visualization
    worldFrustumCorners = getFrustumCornersWorldSpace(invVpMatrix);
  }

  // Write the MAIN camera matrix before the render pass for voxel drawing
  device.queue.writeBuffer(uniformBuffer, 0, vpMatrix as Float32Array);

  // --- Begin Render Pass ---
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

  // --- Draw Voxel Scene ---
  // Voxel scene uses the main camera VP matrix already in the buffer
  const frustumPlanes = extractFrustumPlanes(vpMatrix); // Culling logic uses main matrix
  const sceneStats = drawVoxelScene(
    passEncoder,
    rendererState,
    chunkMeshes,
    frustumPlanes
  );
  rendererState.debugInfo.totalChunks = chunkMeshes.size;
  rendererState.debugInfo.drawnChunks = sceneStats.drawnChunks;
  rendererState.debugInfo.culledChunks = sceneStats.culledChunks;
  rendererState.debugInfo.totalTriangles = sceneStats.totalTriangles;

  let lineData: Float32Array | null = null;
  if (enableDebugView) {
    lineData = generateDebugLineVertices(
      rendererState,
      chunkMeshes,
      frustumPlanes,
      worldFrustumCorners,
      cameraPosition
    );
  }
  if (lineData) {
    // Write the DEBUG camera matrix JUST before drawing lines
    device.queue.writeBuffer(uniformBuffer, 0, vpMatrixDebug as Float32Array);
    drawDebugLines(passEncoder, rendererState, lineData);
    // Restore MAIN camera matrix after drawing lines (good practice)
    device.queue.writeBuffer(
      uniformBuffer,
      0,
      (enableDebugView ? vpMatrixDebug : vpMatrix) as Float32Array
    );
  }

  // --- Finish Up ---
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  return {
    updatedDepthTexture: depthTexture,
    totalTriangles: sceneStats.totalTriangles,
    drawnChunks: sceneStats.drawnChunks,
  };
}
