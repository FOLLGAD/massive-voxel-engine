import { mat4, vec3, vec4 } from "gl-matrix";
import { ChunkManager, type ChunkGeometryInfo } from "./chunk-manager";
import { drawDebugLines, generateDebugLineVertices } from "./renderer.debug";
import { extractFrustumPlanes } from "./renderer.util";
import type { AABB } from "./aabb";
import { getChunkKey, getChunkOfPosition, getPairBitIndex } from "./chunk";

// --- Constants ---
const FRUSTUM_CULLING_EPSILON = 1e-5;
export const DEBUG_COLOR_CULLED = [1, 0, 0]; // Red
export const DEBUG_COLOR_DRAWN = [0, 1, 0]; // Green
export const DEBUG_COLOR_FRUSTUM = [0.8, 0.8, 0]; // Yellow
export const DEBUG_COLOR_PLAYER = [0, 0.8, 0.8]; // Cyan
const INITIAL_DEBUG_LINE_BUFFER_SIZE = 1024 * 6 * 4 * 10 * 100; // ~100k lines
const INITIAL_HIGHLIGHT_BUFFER_SIZE = 1024 * 6; // Enough for ~42 cubes initially
const HIGHLIGHT_COLOR = [1.0, 1.0, 0.0]; // Yellow
const CROSSHAIR_NDC_SIZE = 0.02; // Base size of the crosshair in NDC units (applied vertically)
const CROSSHAIR_COLOR = [1.0, 1.0, 1.0]; // White
const INITIAL_SHARED_VERTEX_BUFFER_SIZE = 1024 * 1024 * 1024; // 1GB
const INITIAL_SHARED_INDEX_BUFFER_SIZE = 256 * 1024 * 1024;  // 256MB
const INDEX_FORMAT: GPUIndexFormat = "uint32";

// @ts-ignore
import voxelShaderCode from "./shaders/voxel.wgsl" with { type: "text" };
// @ts-ignore
import lineShaderCode from "./shaders/line.wgsl" with { type: "text" };
// @ts-ignore
import highlightShaderCode from "./shaders/highlight.wgsl" with { type: "text" };
// @ts-ignore
import cullChunksShader from "./shaders/cullChunks.wgsl" with { type: "text" }; // Unused?
// @ts-ignore
import skyShaderCode from "./shaders/sky.wgsl" with { type: "text" }; // Import the sky shader


// --- Frustum Culling Types ---
/** Represents a plane equation: Ax + By + Cz + D = 0 */
export type Plane = vec4; // [A, B, C, D]

// --- Occlusion Culling Helpers ---

// Face indices (ensure consistency with chunk.ts)
const FACE_X_PLUS = 0;
const FACE_X_MINUS = 1;
const FACE_Y_PLUS = 2;
const FACE_Y_MINUS = 3;
const FACE_Z_PLUS = 4;
const FACE_Z_MINUS = 5;
const NUM_FACES = 6;

// Helper to get the opposite face index
const getOppositeFace = (face: number): number => {
  switch (face) {
    case FACE_X_PLUS: return FACE_X_MINUS;
    case FACE_X_MINUS: return FACE_X_PLUS;
    case FACE_Y_PLUS: return FACE_Y_MINUS;
    case FACE_Y_MINUS: return FACE_Y_PLUS;
    case FACE_Z_PLUS: return FACE_Z_MINUS;
    case FACE_Z_MINUS: return FACE_Z_PLUS;
    default: return -1; // Should not happen
  }
};

// Helper to get the face index corresponding to a neighbor direction delta
// Order: +x, -x, +y, -y, +z, -z
const getFaceFromDirection = (direction: number): number => {
  return direction;
};

// --- Modified cullChunks Function ---

const cullChunks = (
  allChunkInfos: Map<string, ChunkGeometryInfo>,
  frustumPlanes: Plane[],
  cameraPosition: vec3,
  enableAdvancedCulling: boolean
): ChunkGeometryInfo[] => {
  if (!enableAdvancedCulling) {
    return [...allChunkInfos.values()].filter(chunkInfo => Renderer.intersectFrustumAABB(frustumPlanes, chunkInfo.aabb));
  }
  const startChunkPos = getChunkOfPosition(cameraPosition);
  const startChunkKey = getChunkKey(startChunkPos);
  const startChunkInfo = allChunkInfos.get(startChunkKey);

  if (!startChunkInfo) {
    console.error(
      "Start chunk not found",
      startChunkPos,
      startChunkKey,
    )
    // Start chunk doesn't exist, isn't loaded, or is outside frustum
    return [];
  }

  const visibleChunks = new Map<string, ChunkGeometryInfo>(); // Stores potentially visible chunks
  const queue: [ChunkGeometryInfo, number][] = []; // [chunkInfo, entryFaceIndex]
  const visitedKeys = new Set<string>(); // Track keys added to queue to prevent cycles/redundancy

  // Add starting chunk (it passed the frustum check)
  visibleChunks.set(startChunkKey, startChunkInfo);
  queue.push([startChunkInfo, -1]); // -1 indicates no entry face
  visitedKeys.add(startChunkKey);

  const postFrustumCheck = false;

  const neighborChunkPos: vec3 = vec3.create();
  while (queue.length > 0) {
    const [currentChunkInfo, entryFace] = queue.shift() as [ChunkGeometryInfo, number]; // Dequeue

    if (!postFrustumCheck) {
      if (!Renderer.intersectFrustumAABB(frustumPlanes, currentChunkInfo.aabb)) {
        continue;
      }
    }

    const currentChunkPos = currentChunkInfo.position;
    const currentVisibilityBits = currentChunkInfo.visibilityBits;

    // Explore neighbors in 6 directions
    for (let exitDirection = 0; exitDirection < NUM_FACES; exitDirection++) {
      const exitFace = getFaceFromDirection(exitDirection); // Face of current chunk we are exiting through

      // Calculate neighbor position based on direction
      neighborChunkPos[0] = currentChunkPos[0]
      neighborChunkPos[1] = currentChunkPos[1]
      neighborChunkPos[2] = currentChunkPos[2]
      switch (exitDirection) {
        case FACE_X_PLUS: neighborChunkPos[0]++; break;
        case FACE_X_MINUS: neighborChunkPos[0]--; break;
        case FACE_Y_PLUS: neighborChunkPos[1]++; break;
        case FACE_Y_MINUS: neighborChunkPos[1]--; break;
        case FACE_Z_PLUS: neighborChunkPos[2]++; break;
        case FACE_Z_MINUS: neighborChunkPos[2]--; break;
      }

      const neighborChunkKey = getChunkKey(neighborChunkPos);

      // Check if already visited (added to queue or processed)
      if (visitedKeys.has(neighborChunkKey)) {
        continue;
      }

      const neighborChunkInfo = allChunkInfos.get(neighborChunkKey);

      if (!neighborChunkInfo) {
        visitedKeys.add(neighborChunkKey); // Mark as visited even if not loaded to avoid re-checking
        continue; // Neighbor doesn't exist or isn't loaded
      }

      // --- Occlusion Check --- 
      let canSeeNeighbor = false;
      if (entryFace === -1) {
        canSeeNeighbor = true;
        visitedKeys.add(neighborChunkKey);
      } else {
        if (entryFace === exitFace) {
          continue;
        }
        const bitIndex = getPairBitIndex(entryFace, exitFace);
        if (bitIndex !== -1 && (currentVisibilityBits & (1 << bitIndex)) !== 0) {
          canSeeNeighbor = true;
        }
      }

      // --- Visibility Decision --- 
      if (canSeeNeighbor) {
        const neighborEntryFace = getOppositeFace(exitFace);
        queue.push([neighborChunkInfo, neighborEntryFace]);
        visibleChunks.set(neighborChunkKey, neighborChunkInfo);
        visitedKeys.add(neighborChunkKey);
      }
    }
  }

  const visibleChunkInfos = Array.from(visibleChunks.values());
  if (visibleChunkInfos.length === 0) {
    console.log(
      "No visible chunks found",
      startChunkPos,
      frustumPlanes,
      cameraPosition,
      startChunkInfo,
      startChunkKey,
      startChunkInfo.position,
    )
    return [];
  }

  if (postFrustumCheck) {
    return visibleChunkInfos.filter(chunkInfo => Renderer.intersectFrustumAABB(frustumPlanes, chunkInfo.aabb));
  }

  return visibleChunkInfos;
};


// --- Renderer Class ---
export class Renderer {
  public device: GPUDevice;
  public context: GPUCanvasContext;
  public presentationFormat: GPUTextureFormat;
  public voxelPipeline: GPURenderPipeline;
  public linePipeline: GPURenderPipeline;
  public highlightPipeline: GPURenderPipeline;
  public skyPipeline: GPURenderPipeline; // Added sky pipeline
  public uniformBuffer: GPUBuffer;
  public bindGroup: GPUBindGroup;
  public uiUniformBuffer: GPUBuffer;
  public uiBindGroup: GPUBindGroup;
  public skyUniformBuffer: GPUBuffer; // Added sky uniform buffer
  public skyBindGroup: GPUBindGroup;   // Added sky bind group
  public depthTexture: GPUTexture;
  public debugLineBuffer: GPUBuffer;
  public debugLineBufferSize: number;
  public highlightVertexBuffer: GPUBuffer;
  public highlightVertexBufferSize: number;
  public crosshairVertexBuffer: GPUBuffer;
  public crosshairVertexCount: number;
  public skyVertexBuffer: GPUBuffer; // Added sky vertex buffer
  public viewMatrix: mat4;
  public projectionMatrix: mat4;
  public vpMatrix: mat4;
  public viewMatrixDebug: mat4;
  public projectionMatrixDebug: mat4;
  public vpMatrixDebug: mat4;
  public debugInfo: {
    totalChunks: number;
    drawnChunks: number;
    totalTriangles: number;
    culledChunks: number; // This might need recalculation based on the new culling logic
  };
  public chunkManager: ChunkManager;
  public sharedVertexBuffer: GPUBuffer;
  public sharedIndexBuffer: GPUBuffer;
  private canvasWidth = 0;
  private canvasHeight = 0;
  private aspect = 1.0; // Added aspect ratio state

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
    voxelPipeline: GPURenderPipeline,
    linePipeline: GPURenderPipeline,
    highlightPipeline: GPURenderPipeline,
    skyPipeline: GPURenderPipeline, // Added sky pipeline parameter
    uniformBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
    uiUniformBuffer: GPUBuffer,
    uiBindGroup: GPUBindGroup,
    skyUniformBuffer: GPUBuffer, // Added sky uniform buffer parameter
    skyBindGroup: GPUBindGroup,   // Added sky bind group parameter
    debugLineBuffer: GPUBuffer,
    debugLineBufferSize: number,
    highlightVertexBuffer: GPUBuffer,
    crosshairVertexBuffer: GPUBuffer,
    crosshairVertexCount: number,
    skyVertexBuffer: GPUBuffer, // Added sky vertex buffer parameter
    chunkManager: ChunkManager,
    sharedVertexBuffer: GPUBuffer,
    sharedIndexBuffer: GPUBuffer
  ) {
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;
    this.voxelPipeline = voxelPipeline;
    this.linePipeline = linePipeline;
    this.highlightPipeline = highlightPipeline;
    this.skyPipeline = skyPipeline; // Assign sky pipeline
    this.uniformBuffer = uniformBuffer;
    this.bindGroup = bindGroup;
    this.uiUniformBuffer = uiUniformBuffer;
    this.uiBindGroup = uiBindGroup;
    this.skyUniformBuffer = skyUniformBuffer; // Assign sky uniform buffer
    this.skyBindGroup = skyBindGroup;     // Assign sky bind group
    this.debugLineBuffer = debugLineBuffer;
    this.debugLineBufferSize = debugLineBufferSize;
    this.highlightVertexBuffer = highlightVertexBuffer;
    this.highlightVertexBufferSize = INITIAL_HIGHLIGHT_BUFFER_SIZE;
    this.crosshairVertexBuffer = crosshairVertexBuffer;
    this.crosshairVertexCount = crosshairVertexCount;
    this.skyVertexBuffer = skyVertexBuffer; // Assign sky vertex buffer
    this.chunkManager = chunkManager;
    this.sharedVertexBuffer = sharedVertexBuffer;
    this.sharedIndexBuffer = sharedIndexBuffer;

    // Initialize aspect ratio
    this.canvasWidth = this.context.canvas.width;
    this.canvasHeight = this.context.canvas.height;
    this.aspect = this.canvasWidth > 0 && this.canvasHeight > 0 ? this.canvasWidth / this.canvasHeight : 1.0;

    // Matrices
    this.viewMatrix = mat4.create();
    this.projectionMatrix = mat4.create();
    this.vpMatrix = mat4.create();
    this.viewMatrixDebug = mat4.create();
    this.projectionMatrixDebug = mat4.create();
    this.vpMatrixDebug = mat4.create();

    // Debug Info
    this.debugInfo = {
      totalChunks: 0,
      drawnChunks: 0,
      totalTriangles: 0,
      culledChunks: 0,
    };

    // --- Depth Texture ---
    this.depthTexture = this.configureDepthTexture(device, null);
    this.canvasWidth = this.context.canvas.width;
    this.canvasHeight = this.context.canvas.height;
  }

  // --- Static Initialization ---
  public static async create(canvas: HTMLCanvasElement): Promise<Renderer> {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported on this browser.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No appropriate GPUAdapter found.");
    }
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: 1024 * 1024 * 1024 * 2, // 2GB
      }
    });
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

    const uniformBufferSize = (16 + 4 + 4 + 4) * Float32Array.BYTES_PER_ELEMENT;
    const uniformBuffer = device.createBuffer({
      label: "Uniform Buffer (Main Scene VP + Lighting)", // Updated label
      size: uniformBufferSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const sharedBindGroupLayout = device.createBindGroupLayout({ // Renamed for clarity
      label: "Shared Bind Group Layout (Main Uniform)",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    });
    const sharedPipelineLayout = device.createPipelineLayout({ // Renamed for clarity
      label: "Shared Pipeline Layout (Main Uniform)",
      bindGroupLayouts: [sharedBindGroupLayout],
    });
    const bindGroup = device.createBindGroup({
      label: "Main Scene Bind Group",
      layout: sharedBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    // --- UI buffer/group ---
    const uiUniformBuffer = device.createBuffer({
      label: "UI Uniform Buffer (VP Matrix)", // Will hold identity or ortho projection
      size: 64, // Only needs mat4x4 for VP
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // UI needs its own bind group layout if buffer size/structure differs
    const uiBindGroupLayout = device.createBindGroupLayout({
      label: "UI Bind Group Layout",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX,
        buffer: { type: "uniform" },
      }],
    });
    // UI needs its own pipeline layout
    const uiPipelineLayout = device.createPipelineLayout({
      label: "UI Pipeline Layout",
      bindGroupLayouts: [uiBindGroupLayout],
    });
    const uiBindGroup = device.createBindGroup({
      label: "UI Bind Group",
      layout: uiBindGroupLayout, // Use UI layout
      entries: [{ binding: 0, resource: { buffer: uiUniformBuffer } }],
    });

    // --- Sky Resources ---
    const skyUniformBuffer = device.createBuffer({
      label: "Sky Uniform Buffer (Sky VP Matrix)",
      size: 64, // Only needs mat4x4 for VP
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Create a dedicated layout for the skybox (only VP matrix)
    const skyBindGroupLayout = device.createBindGroupLayout({
      label: "Sky Bind Group Layout",
      entries: [{
        binding: 0,
        visibility: GPUShaderStage.VERTEX, // Only vertex shader needs VP
        buffer: { type: "uniform" },
      }],
    });
    const skyPipelineLayout = device.createPipelineLayout({
      label: "Sky Pipeline Layout",
      bindGroupLayouts: [skyBindGroupLayout],
    });
    // Sky bind group uses the dedicated sky layout
    const skyBindGroup = device.createBindGroup({
      label: "Sky Bind Group",
      layout: skyBindGroupLayout, // Use dedicated sky layout
      entries: [{ binding: 0, resource: { buffer: skyUniformBuffer } }],
    });

    // --- Pipelines ---
    // Voxel, Line use sharedPipelineLayout
    const voxelPipeline = Renderer.createVoxelPipeline(device, presentationFormat, sharedPipelineLayout);
    const linePipeline = Renderer.createLinePipeline(device, presentationFormat, sharedPipelineLayout);
    // Highlight uses uiPipelineLayout as it typically doesn't need lighting etc.
    const highlightPipeline = Renderer.createHighlightPipeline(device, presentationFormat, uiPipelineLayout); // Use UI layout
    // Sky uses its dedicated skyPipelineLayout
    const skyPipeline = Renderer.createSkyPipeline(device, presentationFormat, skyPipelineLayout); // Use dedicated sky layout


    // --- Create Shared Buffers ---
    const sharedVertexBuffer = device.createBuffer({
      label: "Shared Vertex Buffer",
      size: INITIAL_SHARED_VERTEX_BUFFER_SIZE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    const sharedIndexBuffer = device.createBuffer({
      label: "Shared Index Buffer",
      size: INITIAL_SHARED_INDEX_BUFFER_SIZE,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    });

    // --- Create Chunk Manager ---
    const chunkManager = new ChunkManager(device, sharedVertexBuffer, sharedIndexBuffer);

    // --- Buffers (Debug, Highlight, Crosshair, Sky) ---
    const debugLineBuffer = device.createBuffer({
      label: "Debug Line Buffer",
      size: INITIAL_DEBUG_LINE_BUFFER_SIZE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const highlightVertexBuffer = device.createBuffer({
      label: "Highlight Vertex Buffer",
      size: INITIAL_HIGHLIGHT_BUFFER_SIZE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    const aspect = canvas.width > 0 && canvas.height > 0 ? canvas.width / canvas.height : 1.0;
    const crosshairVertices = Renderer.calculateCrosshairVertices(aspect);
    const crosshairVertexCount = 4; // 2 lines * 2 vertices
    const crosshairVertexBuffer = device.createBuffer({
      label: "Crosshair Vertex Buffer",
      size: crosshairVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true, // Map initially to write data
    });
    new Float32Array(crosshairVertexBuffer.getMappedRange()).set(crosshairVertices);
    crosshairVertexBuffer.unmap();


    // Sky Vertex Buffer (Unit Cube)
    const skyVertexData = Renderer.createCubeVertices();
    const skyVertexBuffer = device.createBuffer({
      label: "Sky Vertex Buffer",
      size: skyVertexData.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(skyVertexBuffer.getMappedRange()).set(skyVertexData);
    skyVertexBuffer.unmap();


    // Create the Renderer instance
    const renderer = new Renderer(
      device,
      context,
      presentationFormat,
      voxelPipeline,
      linePipeline,
      highlightPipeline,
      skyPipeline, // Pass sky pipeline
      uniformBuffer,
      bindGroup,
      uiUniformBuffer, // Pass UI buffer
      uiBindGroup, // Pass UI bind group
      skyUniformBuffer, // Pass sky uniform buffer
      skyBindGroup,   // Pass sky bind group
      debugLineBuffer,
      INITIAL_DEBUG_LINE_BUFFER_SIZE,
      highlightVertexBuffer,
      crosshairVertexBuffer,
      crosshairVertexCount,
      skyVertexBuffer, // Pass sky vertex buffer
      chunkManager,
      sharedVertexBuffer,
      sharedIndexBuffer
    );

    // Write initial UI matrix (identity) *after* Renderer creation
    renderer.device.queue.writeBuffer(renderer.uiUniformBuffer, 0, mat4.create() as Float32Array);

    // No need to update crosshair buffer here, it was written at creation

    return renderer;
  }

  private static createCubeVertices(): Float32Array {
    // Correct CCW winding for rendering from inside
    return new Float32Array([
      // Front face (+Z)
      -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, // Triangle 1
      -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, // Triangle 2

      // Back face (-Z)
      1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, // Triangle 3
      1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, // Triangle 4

      // Left face (-X)
      -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, // Triangle 5
      -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, // Triangle 6

      // Right face (+X)
      1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, // Triangle 7
      1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, // Triangle 8

      // Top face (+Y)
      -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, // Triangle 9
      -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, // Triangle 10

      // Bottom face (-Y)
      -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, // Triangle 11
      -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, // Triangle 12
    ]);
  }



  // --- Pipeline Creation ---
  private static createVoxelPipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout // Shared layout (VP + Lighting)
  ): GPURenderPipeline {
    const voxelShaderModule = device.createShaderModule({
      code: voxelShaderCode,
    });
    const voxelVertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 9 * Float32Array.BYTES_PER_ELEMENT, // 36 bytes: pos(3) + color(3) + normal(3)
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

  private static createLinePipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout // Shared layout (VP + Lighting - though lighting often unused for lines)
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
      layout: pipelineLayout, // Use shared layout
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
        topology: "line-list",
      },
      depthStencil: {
        depthWriteEnabled: true, // Write depth for debug lines usually
        depthCompare: "less",
        format: "depth24plus",
      },
    });
  }

  private static createHighlightPipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout // UI layout (only VP)
  ): GPURenderPipeline {
    const highlightShaderModule = device.createShaderModule({
      code: highlightShaderCode, // Assumes a simple shader taking pos, color, and VP matrix
    });

    const highlightVertexBufferLayout: GPUVertexBufferLayout = {
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
      label: "Highlight Render Pipeline",
      layout: pipelineLayout, // Use UI layout
      vertex: {
        module: highlightShaderModule,
        entryPoint: "vs_main",
        buffers: [highlightVertexBufferLayout],
      },
      fragment: {
        module: highlightShaderModule,
        entryPoint: "fs_main",
        targets: [{ format: presentationFormat }],
      },
      primitive: {
        topology: "line-list",
      },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: "less",
        format: "depth24plus",
      },
    });
  }

  private static createSkyPipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout // Shared layout (VP matrix at binding 0)
  ): GPURenderPipeline {
    const skyShaderModule = device.createShaderModule({
      code: skyShaderCode,
    });

    const skyVertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT, // Only position
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
      ],
    };

    return device.createRenderPipeline({
      label: "Sky Render Pipeline",
      layout: pipelineLayout, // Use shared layout (assumes binding 0 is VP)
      vertex: {
        module: skyShaderModule,
        entryPoint: "vs_main",
        buffers: [skyVertexBufferLayout],
      },
      fragment: {
        module: skyShaderModule,
        entryPoint: "fs_main",
        targets: [{ format: presentationFormat }],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "front", // Cull front faces because we are inside the cube
      },
      depthStencil: {
        depthWriteEnabled: false, // Don't write depth
        depthCompare: "less-equal", // Draw sky where depth is 1.0 (or equal for edge cases)
        format: "depth24plus",
      },
    });
  }

  // --- Helper Methods ---

  private static updateViewMatrix(
    viewMatrix: mat4,
    eye: vec3,
    pitch: number,
    yaw: number,
    up: vec3 = vec3.fromValues(0, 1, 0)
  ) {
    const direction = vec3.create();
    // Ensure pitch is clamped to avoid gimbal lock issues
    const clampedPitch = Math.max(-Math.PI / 2 + 1e-6, Math.min(Math.PI / 2 - 1e-6, pitch));
    direction[0] = Math.cos(clampedPitch) * Math.sin(yaw);
    direction[1] = Math.sin(clampedPitch);
    direction[2] = Math.cos(clampedPitch) * Math.cos(yaw);

    const center = vec3.create();
    vec3.add(center, eye, direction);

    mat4.lookAt(viewMatrix, eye, center, up);
  }

  // Helper to calculate aspect-corrected crosshair vertices
  private static calculateCrosshairVertices(aspect: number): Float32Array {
    const sizeY = CROSSHAIR_NDC_SIZE;
    const sizeX = CROSSHAIR_NDC_SIZE / aspect; // Scale X by inverse aspect ratio

    // 2 lines, 4 vertices total. Each vertex: x, y, z, r, g, b
    return new Float32Array([
      // Horizontal line
      -sizeX, 0.0, 0.0, ...CROSSHAIR_COLOR, // Left point
      sizeX, 0.0, 0.0, ...CROSSHAIR_COLOR, // Right point
      // Vertical line
      0.0, -sizeY, 0.0, ...CROSSHAIR_COLOR, // Bottom point
      0.0, sizeY, 0.0, ...CROSSHAIR_COLOR,  // Top point
    ]);
  }

  // Helper to update the crosshair vertex buffer
  private updateCrosshairBuffer(aspect: number): void {
    const vertices = Renderer.calculateCrosshairVertices(aspect);
    this.device.queue.writeBuffer(this.crosshairVertexBuffer, 0, vertices);
  }

  private configureDepthTexture(
    device: GPUDevice,
    currentDepthTexture: GPUTexture | null
  ): GPUTexture {
    if (
      currentDepthTexture &&
      currentDepthTexture.width === this.canvasWidth &&
      currentDepthTexture.height === this.canvasHeight
    ) {
      return currentDepthTexture; // Reuse existing texture if size matches
    }
    // Destroy previous texture if it exists
    if (currentDepthTexture) {
      currentDepthTexture.destroy();
    }
    // Create new texture with current canvas dimensions
    return device.createTexture({
      label: "Depth Texture",
      size: [this.canvasWidth, this.canvasHeight],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  /**
   * Checks if an AABB intersects with the view frustum.
   * @param planes The 6 planes defining the frustum.
   * @param aabb The Axis-Aligned Bounding Box to check.
   * @returns True if the AABB intersects or is contained within the frustum, false otherwise.
   */
  public static intersectFrustumAABB(
    planes: Plane[],
    aabb: AABB
  ): boolean {
    for (let i = 0; i < 6; i++) {
      const plane = planes[i]; // Plane normal (A, B, C) and distance D

      // Find the vertex of the AABB that is furthest in the *negative* direction of the plane normal (p-vertex)
      // and the vertex furthest in the *positive* direction (n-vertex).
      // We only need the positive vertex (n-vertex) for the standard check.
      // The n-vertex is the one where each coordinate (x, y, z) is chosen
      // from the AABB's min or max corner based on the sign of the corresponding
      // component of the plane's normal.
      const positiveVertex: vec3 = [
        plane[0] > 0 ? aabb.max[0] : aabb.min[0],
        plane[1] > 0 ? aabb.max[1] : aabb.min[1],
        plane[2] > 0 ? aabb.max[2] : aabb.min[2],
      ];

      // Calculate the signed distance from the positive vertex to the plane.
      // distance = Ax + By + Cz + D
      const distance = vec4.dot(plane, vec4.fromValues(positiveVertex[0], positiveVertex[1], positiveVertex[2], 1.0));

      // If the positive vertex is behind the plane (distance < 0),
      // the entire AABB must be outside this plane, and thus outside the frustum.
      // Use a small epsilon for floating-point comparisons.
      if (distance < -FRUSTUM_CULLING_EPSILON) {
        return false; // AABB is completely outside this plane
      }

      // Optimization: We could also check the negative vertex (p-vertex)
      // If `dot(plane, p-vertex) > 0`, the AABB is fully *inside* this plane.
      // If the AABB is fully inside *all* planes, it's fully contained in the frustum.
      // However, for simple culling, just checking the positive vertex is sufficient.
    }
    // If the AABB is not completely outside any single plane, it must intersect the frustum.
    return true;
  }


  private static getFrustumCornersWorldSpace(invVpMatrix: mat4): vec3[] {
    const ndcCorners: vec4[] = [
      [-1, -1, -1, 1], [-1, 1, -1, 1], [1, 1, -1, 1], [1, -1, -1, 1], // Near plane corners
      [-1, -1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1], [1, -1, 1, 1],   // Far plane corners
    ].map(v => vec4.fromValues(v[0], v[1], v[2], v[3]));

    return ndcCorners.map(ndc => {
      const worldVec4 = vec4.transformMat4(vec4.create(), ndc, invVpMatrix);
      // Perform perspective divide (w-divide)
      if (worldVec4[3] !== 0) {
        vec4.scale(worldVec4, worldVec4, 1.0 / worldVec4[3]);
      }
      return vec3.fromValues(worldVec4[0], worldVec4[1], worldVec4[2]);
    });
  }

  // --- Drawing Methods ---

  private drawVoxelScene(
    passEncoder: GPURenderPassEncoder,
    visibleChunkInfos: ChunkGeometryInfo[]
  ): { drawnChunks: number; totalTriangles: number } {
    if (visibleChunkInfos.length === 0) {
      return { drawnChunks: 0, totalTriangles: 0 };
    }

    passEncoder.setPipeline(this.voxelPipeline);
    passEncoder.setBindGroup(0, this.bindGroup); // Use main bind group (VP + Lighting)

    // Set the shared buffers ONCE before the loop
    passEncoder.setVertexBuffer(0, this.sharedVertexBuffer);
    passEncoder.setIndexBuffer(this.sharedIndexBuffer, INDEX_FORMAT); // Use constant for format

    let totalTriangles = 0;
    let drawnChunks = 0;

    // Draw all visible chunks using offsets
    for (const info of visibleChunkInfos) {
        passEncoder.drawIndexed(info.indexCount, 1, info.firstIndex, info.baseVertex, 0);
        totalTriangles += info.indexCount / 3;
        drawnChunks++;
    }

    // Return drawn count and triangles based on the loops
    return { drawnChunks, totalTriangles };
  }

  // --- Render Frame ---
  public renderFrame(
    cameraPosition: vec3,
    cameraPitch: number,
    cameraYaw: number,
    highlightedBlockPositions: vec3[],
    fov: number, // Field of View in radians
    debugCamera?: {
      position: vec3,
      target: vec3
    },
    enableDebugView = true,
    enableAdvancedCulling = false
  ): { totalTriangles: number; drawnChunks: number } {
    // Use the potentially updated aspect ratio
    const currentAspect = this.aspect; // Use internal aspect state

    // --- Depth Texture ---
    // Ensure depth texture matches current canvas size (configureDepthTexture handles this)
    this.depthTexture = this.configureDepthTexture(this.device, this.depthTexture);


    // --- Calculate Matrices ---
    // Main camera matrices
    Renderer.updateViewMatrix(this.viewMatrix, cameraPosition, cameraPitch, cameraYaw);
    mat4.perspective(this.projectionMatrix, fov, currentAspect, 0.1, 1000.0); // Use current aspect
    mat4.multiply(this.vpMatrix, this.projectionMatrix, this.viewMatrix);

    // Sky View-Projection Matrix (uses main projection, but view matrix with no translation)
    const skyViewMatrix = mat4.clone(this.viewMatrix);
    skyViewMatrix[12] = 0; // Zero out translation X
    skyViewMatrix[13] = 0; // Zero out translation Y
    skyViewMatrix[14] = 0; // Zero out translation Z
    const skyVpMatrix = mat4.create();
    mat4.multiply(skyVpMatrix, this.projectionMatrix, skyViewMatrix);

    // Debug camera matrices (if enabled)
    const invVpMatrix = mat4.invert(mat4.create(), this.vpMatrix); // Inverse of *main* VP
    let worldFrustumCorners: vec3[] = []; // Corners of the main camera frustum
    let activeVpMatrix = this.vpMatrix; // VP matrix used for rendering the main scene

    if (debugCamera) {
      mat4.lookAt(
        this.viewMatrixDebug,
        debugCamera.position,
        debugCamera.target,
        vec3.fromValues(0, 1, 0)
      );
      // Use the same projection as the main camera for the debug view
      mat4.copy(this.projectionMatrixDebug, this.projectionMatrix);
      mat4.multiply(this.vpMatrixDebug, this.projectionMatrixDebug, this.viewMatrixDebug);

      // Calculate the main frustum corners in world space for drawing
      worldFrustumCorners = Renderer.getFrustumCornersWorldSpace(invVpMatrix);
      // Set the active VP matrix to the debug one for rendering voxels/lines
      activeVpMatrix = this.vpMatrixDebug;
    }

    // --- Write Uniform Buffers (Before Render Pass) ---

    // Main Uniform Buffer (VP + Lighting)
    const uniformData = new Float32Array(24); // 16 MVP + 4 LightDir + 4 LightColor + 4 Ambient + pad? = 24 floats
    // Set VP matrix (either normal or debug) at offset 0 (mat4 = 16 floats)
    uniformData.set(activeVpMatrix);

    // Define light data (should ideally come from scene state)
    const lightDirection = vec3.normalize(vec3.create(), [0.8, 0.6, 0.2]);
    const lightColor = [1.0, 1.0, 0.5]; // Slightly yellowish sunlight
    const ambientIntensity = 0.7;

    // Set light direction (offset 64 bytes / 4 = 16 floats) - vec3 padded to vec4 in buffer
    uniformData.set(lightDirection, 16);
    // Set light color (offset 80 bytes / 4 = 20 floats) - vec3 padded to vec4 in buffer
    uniformData.set(lightColor, 20);
    // Set ambient intensity (offset 96 bytes / 4 = 24 floats) - f32
    uniformData[23] = ambientIntensity;

    // Write the combined data to the main uniform buffer
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Sky Uniform Buffer (only VP matrix for sky)
    this.device.queue.writeBuffer(this.skyUniformBuffer, 0, skyVpMatrix as Float32Array);

    // UI Uniform Buffer (Identity matrix for crosshair, VP for highlights)
    const identityMatrix = mat4.create(); // For crosshair
    this.device.queue.writeBuffer(this.uiUniformBuffer, 0, identityMatrix as Float32Array); // Write identity for now


    // --- Begin Render Pass ---
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    const passEncoder = commandEncoder.beginRenderPass({
      label: "Main Render Pass",
      colorAttachments: [
        {
          view: textureView,
          // Clear color is less important if skybox covers everything, but good practice
          clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear", // Clear the color buffer at the start
          storeOp: "store", // Store the result
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0, // Important: Clear depth to max distance
        depthLoadOp: "clear", // Clear the depth buffer at the start
        depthStoreOp: "store", // Store depth values for subsequent draws
      },
    });

    // --- Draw Sky FIRST ---
    passEncoder.setPipeline(this.skyPipeline);
    passEncoder.setBindGroup(0, this.skyBindGroup);
    passEncoder.setVertexBuffer(0, this.skyVertexBuffer);
    passEncoder.draw(36);


    // --- Cull Chunks and Prepare Visible List ---
    const frustumPlanes = extractFrustumPlanes(this.vpMatrix);
    const visibleChunks = cullChunks(
      this.chunkManager.chunkGeometryInfo,
      frustumPlanes,
      cameraPosition,
      enableAdvancedCulling
    );
    const totalChunks = this.chunkManager.chunkGeometryInfo.size;
    const culledChunkCount = totalChunks - visibleChunks.length;


    // --- Draw Voxel Scene ---
    // Voxel scene uses the 'activeVpMatrix' (normal or debug) via the main uniform buffer/bind group
    const sceneStats = this.drawVoxelScene(passEncoder, visibleChunks);

    // Update debug info (more accurate now)
    this.debugInfo.totalChunks = totalChunks;
    this.debugInfo.drawnChunks = sceneStats.drawnChunks;
    this.debugInfo.culledChunks = culledChunkCount; // Calculated during culling
    this.debugInfo.totalTriangles = sceneStats.totalTriangles;


    // --- Prepare and Draw Highlights ---
    // Highlights use the 'activeVpMatrix' but via the UI bind group/buffer
    let totalHighlightVertices = 0;
    if (highlightedBlockPositions.length > 0) {
      const numberOfHighlightedCubes = highlightedBlockPositions.length;
      const highlightVertexData = this.generateHighlightVertices(highlightedBlockPositions);
      totalHighlightVertices = numberOfHighlightedCubes * 24; // 24 vertices per cube (12 lines * 2 vertices)

      // Resize buffer if needed
      if (highlightVertexData.byteLength > this.highlightVertexBufferSize) {
        this.highlightVertexBuffer.destroy();
        this.highlightVertexBufferSize = Math.max(this.highlightVertexBufferSize * 2, highlightVertexData.byteLength);
        this.highlightVertexBuffer = this.device.createBuffer({
          label: "Highlight Vertex Buffer (Resized)",
          size: this.highlightVertexBufferSize,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        console.warn("Resized highlight vertex buffer to:", this.highlightVertexBufferSize);
      }
      this.device.queue.writeBuffer(this.highlightVertexBuffer, 0, highlightVertexData);

      // Update the UI buffer with the *active* VP matrix for highlights
      this.device.queue.writeBuffer(this.uiUniformBuffer, 0, activeVpMatrix as Float32Array);

      passEncoder.setPipeline(this.highlightPipeline);
      passEncoder.setBindGroup(0, this.uiBindGroup); // Use UI bind group (now holds active VP)
      passEncoder.setVertexBuffer(0, this.highlightVertexBuffer);
      passEncoder.draw(totalHighlightVertices, 1, 0, 0); // Draw all highlight vertices
    }


    // --- Draw Debug Lines ---
    // Debug lines use the 'activeVpMatrix' via the main uniform buffer/bind group
    if (enableDebugView) {
      // Uniform buffer (main) already contains activeVpMatrix

      const lineData = generateDebugLineVertices(
        this.chunkManager,
        frustumPlanes, // Color/cull based on main frustum
        debugCamera ? worldFrustumCorners : [], // Draw main frustum corners only if debug view active
        cameraPosition, // Player position for coloring chunks
      );

      // Resize debug line buffer if necessary
      if (lineData.byteLength > this.debugLineBufferSize) {
        this.debugLineBuffer.destroy();
        this.debugLineBufferSize = Math.max(this.debugLineBufferSize * 2, lineData.byteLength);
        this.debugLineBuffer = this.device.createBuffer({
          label: "Debug Line Buffer (Resized)",
          size: this.debugLineBufferSize,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        console.warn("Resized debug line buffer to:", this.debugLineBufferSize);
      }
      if (lineData.byteLength > 0) {
        this.device.queue.writeBuffer(this.debugLineBuffer, 0, lineData);

        // drawDebugLines uses the Renderer instance to get the linePipeline etc.
        // It should use the main bind group (binding 0) which holds the active VP matrix
        drawDebugLines(passEncoder, this, lineData, this.vpMatrix); // Pass vertex count
      }
    }

    // --- Draw Crosshair ---
    // Crosshair uses the UI uniform buffer/bind group (identity matrix)
    // Make sure UI buffer has identity matrix again if highlights changed it
    this.device.queue.writeBuffer(this.uiUniformBuffer, 0, identityMatrix as Float32Array);

    passEncoder.setPipeline(this.highlightPipeline); // Use highlight pipe (no depth write, simple shader)
    passEncoder.setBindGroup(0, this.uiBindGroup);   // Use UI bind group (identity matrix)
    passEncoder.setVertexBuffer(0, this.crosshairVertexBuffer);
    passEncoder.draw(this.crosshairVertexCount); // Draw 4 vertices


    // --- Finish Up ---
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Return overall stats
    return {
      totalTriangles: sceneStats.totalTriangles,
      drawnChunks: sceneStats.drawnChunks,
      // Consider returning culledChunks as well if needed externally
    };
  }

  // Method to resize canvas and internal textures
  public resize(width: number, height: number) {
    const newWidth = Math.max(1, Math.floor(width)); // Ensure positive integer dimensions
    const newHeight = Math.max(1, Math.floor(height));

    // Check if dimensions actually changed
    if (this.canvasWidth === newWidth && this.canvasHeight === newHeight) {
      return; // No change needed
    }

    console.log(`Renderer resizing from ${this.canvasWidth}x${this.canvasHeight} to ${newWidth}x${newHeight}`);

    // Update internal dimensions
    this.canvasWidth = newWidth;
    this.canvasHeight = newHeight;

    // Update canvas presentation size (consider device pixel ratio for sharpness)
    const canvas = this.context.canvas as HTMLCanvasElement;
    // canvas.width = newWidth * window.devicePixelRatio; // Optional: Adjust for device pixel ratio
    // canvas.height = newHeight * window.devicePixelRatio;
    canvas.width = newWidth; // Keep it simple for now
    canvas.height = newHeight;
    // canvas.style.width = `${newWidth}px`;
    // canvas.style.height = `${newHeight}px`;


    // Reconfigure depth texture (this handles creation/destruction)
    this.depthTexture = this.configureDepthTexture(this.device, this.depthTexture);

    // Update aspect ratio and dependent elements (e.g., crosshair)
    this.aspect = this.canvasWidth / this.canvasHeight;
    this.updateCrosshairBuffer(this.aspect);

    // Note: Projection matrix will be updated in the next renderFrame call using the new aspect ratio

    console.log(`Renderer resized to ${this.canvasWidth}x${this.canvasHeight}, aspect: ${this.aspect.toFixed(2)}`);
  }

  // --- Getters for internal state if needed ---
  public getDevice(): GPUDevice {
    return this.device;
  }

  public getContext(): GPUCanvasContext {
    return this.context;
  }

  public getPresentationFormat(): GPUTextureFormat {
    return this.presentationFormat;
  }

  public getDebugLineBuffer(): GPUBuffer {
    return this.debugLineBuffer;
  }

  public getDebugLineBufferSize(): number {
    return this.debugLineBufferSize;
  }

  public getLinePipeline(): GPURenderPipeline {
    return this.linePipeline; // Needed by drawDebugLines
  }

  public getMainBindGroup(): GPUBindGroup {
    return this.bindGroup; // Needed by drawDebugLines
  }


  // Helper to generate wireframe cube vertices for highlighting
  private generateHighlightVertices(positions: vec3[]): Float32Array {
    const verticesPerCube = 24; // 12 lines * 2 vertices per line
    const floatsPerVertex = 6; // 3 pos, 3 color
    const totalFloats = positions.length * verticesPerCube * floatsPerVertex;
    const data = new Float32Array(totalFloats);
    let offset = 0;

    // Define the 12 edges of a unit cube (0,0,0) to (1,1,1)
    const cubeEdges = [
      // Bottom face
      [0, 0, 0], [1, 0, 0],
      [1, 0, 0], [1, 0, 1],
      [1, 0, 1], [0, 0, 1],
      [0, 0, 1], [0, 0, 0],
      // Top face
      [0, 1, 0], [1, 1, 0],
      [1, 1, 0], [1, 1, 1],
      [1, 1, 1], [0, 1, 1],
      [0, 1, 1], [0, 1, 0],
      // Connecting edges
      [0, 0, 0], [0, 1, 0],
      [1, 0, 0], [1, 1, 0],
      [1, 0, 1], [1, 1, 1],
      [0, 0, 1], [0, 1, 1],
    ];

    for (const pos of positions) {
      for (let i = 0; i < cubeEdges.length; i += 2) {
        const startOffset = cubeEdges[i];
        const endOffset = cubeEdges[i + 1];

        // Start vertex of the edge
        data[offset++] = pos[0] + startOffset[0];
        data[offset++] = pos[1] + startOffset[1];
        data[offset++] = pos[2] + startOffset[2];
        data[offset++] = HIGHLIGHT_COLOR[0];
        data[offset++] = HIGHLIGHT_COLOR[1];
        data[offset++] = HIGHLIGHT_COLOR[2];

        // End vertex of the edge
        data[offset++] = pos[0] + endOffset[0];
        data[offset++] = pos[1] + endOffset[1];
        data[offset++] = pos[2] + endOffset[2];
        data[offset++] = HIGHLIGHT_COLOR[0];
        data[offset++] = HIGHLIGHT_COLOR[1];
        data[offset++] = HIGHLIGHT_COLOR[2];
      }
    }
    return data;
  }
}
