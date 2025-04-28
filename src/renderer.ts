import { mat4, vec3, vec4 } from "gl-matrix";
import { ChunkManager, type ChunkGeometryInfo } from "./chunk-manager";
import { drawDebugLines, generateDebugLineVertices } from "./renderer.debug";
import { extractFrustumPlanes } from "./renderer.util";
import type { AABB } from "./aabb";

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
import cullChunksShader from "./shaders/cullChunks.wgsl" with { type: "text" };
// @ts-ignore
import skyShaderCode from "./shaders/sky.wgsl" with { type: "text" }; // Import the sky shader

// --- Frustum Culling Types ---
/** Represents a plane equation: Ax + By + Cz + D = 0 */
export type Plane = vec4; // [A, B, C, D]

const ENABLE_CHUNK_DEBUG_LINES = false;

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
    culledChunks: number;
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

    // --- Uniform Buffer, Bind Group, Layout (Shared by Voxel, Line, Highlight) ---
    const uniformBufferSize = (16 + 4 + 4 + 4) * Float32Array.BYTES_PER_ELEMENT; // MVP (16) + LightDir(vec3+pad=4) + LightColor(vec3+pad=4) + Ambient(f32=1) + Padding = 28? NO -> 16+3+1+3+1+1+1 = 26? NO 96 bytes -> 24 floats
    const uniformBuffer = device.createBuffer({
      label: "Uniform Buffer (Main Scene MVP Matrix + Lighting)", // Updated label
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

    // --- UI buffer/group (uses same shared layout) ---
    const uiUniformBuffer = device.createBuffer({
        label: "UI Uniform Buffer (Identity Matrix)",
        size: uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const uiBindGroup = device.createBindGroup({
        label: "UI Bind Group",
        layout: sharedBindGroupLayout, // Reuses the main layout
        entries: [{ binding: 0, resource: { buffer: uiUniformBuffer } }],
    });

    // --- Sky Resources ---
    const skyUniformBuffer = device.createBuffer({
        label: "Sky Uniform Buffer (Sky VP Matrix)",
        size: uniformBufferSize, // Same size (mat4)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Sky bind group layout is the same as shared one (just 1 uniform buffer)
    const skyBindGroup = device.createBindGroup({
        label: "Sky Bind Group",
        layout: sharedBindGroupLayout, // Reuses the main layout
        entries: [{ binding: 0, resource: { buffer: skyUniformBuffer } }],
    });

    // --- Pipelines ---
    // Shared pipeline layout used for Voxel, Line, Highlight, and Sky
    const voxelPipeline = Renderer.createVoxelPipeline(device, presentationFormat, sharedPipelineLayout);
    const linePipeline = Renderer.createLinePipeline(device, presentationFormat, sharedPipelineLayout);
    const highlightPipeline = Renderer.createHighlightPipeline(device, presentationFormat, sharedPipelineLayout);
    const skyPipeline = Renderer.createSkyPipeline(device, presentationFormat, sharedPipelineLayout); // Create sky pipeline

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
    const crosshairVertexCount = 4;
    const crosshairVertexBuffer = device.createBuffer({
        label: "Crosshair Vertex Buffer",
        size: crosshairVertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

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
      uiUniformBuffer,
      uiBindGroup,
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

    // Write initial crosshair data *after* Renderer creation
    renderer.updateCrosshairBuffer(aspect);

    return renderer;
  }

  private static createCubeVertices(): Float32Array {
    // Define cube vertices explicitly per triangle (CCW from inside)
    return new Float32Array([
        // Front face (+Z)
        -1.0, -1.0,  1.0,   1.0, -1.0,  1.0,   1.0,  1.0,  1.0, // Triangle 1
        -1.0, -1.0,  1.0,   1.0,  1.0,  1.0,  -1.0,  1.0,  1.0, // Triangle 2

        // Back face (-Z)
         1.0, -1.0, -1.0,  -1.0, -1.0, -1.0,  -1.0,  1.0, -1.0, // Triangle 3
         1.0, -1.0, -1.0,  -1.0,  1.0, -1.0,   1.0,  1.0, -1.0, // Triangle 4

        // Left face (-X)
        -1.0, -1.0, -1.0,  -1.0, -1.0,  1.0,  -1.0,  1.0,  1.0, // Triangle 5
        -1.0, -1.0, -1.0,  -1.0,  1.0,  1.0,  -1.0,  1.0, -1.0, // Triangle 6

        // Right face (+X)
         1.0, -1.0,  1.0,   1.0, -1.0, -1.0,   1.0,  1.0, -1.0, // Triangle 7
         1.0, -1.0,  1.0,   1.0,  1.0, -1.0,   1.0,  1.0,  1.0, // Triangle 8

        // Top face (+Y)
        -1.0,  1.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0, -1.0, // Triangle 9
        -1.0,  1.0,  1.0,   1.0,  1.0, -1.0,  -1.0,  1.0, -1.0, // Triangle 10

        // Bottom face (-Y)
        -1.0, -1.0, -1.0,   1.0, -1.0, -1.0,   1.0, -1.0,  1.0, // Triangle 11
        -1.0, -1.0, -1.0,   1.0, -1.0,  1.0,  -1.0, -1.0,  1.0, // Triangle 12
    ]);
  }

  

  // --- Pipeline Creation ---
  private static createVoxelPipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout
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
      layout: pipelineLayout,
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
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
      },
    });
  }

  private static createHighlightPipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout
  ): GPURenderPipeline {
    const highlightShaderModule = device.createShaderModule({
      code: highlightShaderCode,
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
      layout: pipelineLayout,
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
        topology: "line-list", // Use line-list for wireframe
      },
      depthStencil: {
        depthWriteEnabled: false, // Don't write to depth buffer
        depthCompare: "less",
        format: "depth24plus",
      },
    });
  }

  private static createSkyPipeline(
    device: GPUDevice,
    presentationFormat: GPUTextureFormat,
    pipelineLayout: GPUPipelineLayout // Reuse shared layout
  ): GPURenderPipeline {
    const skyShaderModule = device.createShaderModule({
      code: skyShaderCode,
    });

    const skyVertexBufferLayout: GPUVertexBufferLayout = {
      arrayStride: 3 * Float32Array.BYTES_PER_ELEMENT,
      attributes: [
        { shaderLocation: 0, offset: 0, format: "float32x3" },
      ],
    };

    return device.createRenderPipeline({
      label: "Sky Render Pipeline",
      layout: pipelineLayout,
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
        cullMode: "front",
      },
      depthStencil: {
        depthWriteEnabled: false,
        depthCompare: "less-equal",
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
    direction[0] = Math.cos(pitch) * Math.sin(yaw);
    direction[1] = Math.sin(pitch);
    direction[2] = Math.cos(pitch) * Math.cos(yaw);
    vec3.normalize(direction, direction);

    const center = vec3.create();
    vec3.add(center, eye, direction);

    mat4.lookAt(viewMatrix, eye, center, up);
  }

  // Helper to calculate aspect-corrected crosshair vertices
  private static calculateCrosshairVertices(aspect: number): Float32Array {
    const sizeY = CROSSHAIR_NDC_SIZE;
    const sizeX = CROSSHAIR_NDC_SIZE / aspect; // Scale X by inverse aspect ratio

    return new Float32Array([
        // Horizontal line
        -sizeX, 0.0, 0.0, ...CROSSHAIR_COLOR, // Left point
         sizeX, 0.0, 0.0, ...CROSSHAIR_COLOR, // Right point
        // Vertical line
         0.0, -sizeY, 0.0, ...CROSSHAIR_COLOR, // Bottom point
         0.0,  sizeY, 0.0, ...CROSSHAIR_COLOR,  // Top point
    ]);
  }

  // Helper to update the crosshair vertex buffer
  private updateCrosshairBuffer(aspect: number): void {
      const vertices = Renderer.calculateCrosshairVertices(aspect);
      // Ensure buffer exists and has the correct usage flags
      if (this.crosshairVertexBuffer && (this.crosshairVertexBuffer.usage & GPUBufferUsage.COPY_DST)) {
          this.device.queue.writeBuffer(this.crosshairVertexBuffer, 0, vertices);
      } else {
          console.error("Crosshair vertex buffer is not available or cannot be written to.");
          // Optionally, recreate the buffer here if it's missing or misconfigured
      }
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
      return currentDepthTexture;
    }
    if (currentDepthTexture) {
      currentDepthTexture.destroy();
    }
    return device.createTexture({
      label: "Depth Texture",
      size: [this.canvasWidth, this.canvasHeight],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
  }

  public static intersectFrustumAABB(
    planes: Plane[],
    aabb: AABB
  ): boolean {
    for (let i = 0; i < 6; i++) {
      const plane = planes[i];
      const positiveVertex: vec3 = [
        plane[0] > 0 ? aabb.max[0] : aabb.min[0],
        plane[1] > 0 ? aabb.max[1] : aabb.min[1],
        plane[2] > 0 ? aabb.max[2] : aabb.min[2],
      ];
      const distance = vec4.dot(plane, vec4.fromValues(positiveVertex[0], positiveVertex[1], positiveVertex[2], 1.0));
      if (distance < -FRUSTUM_CULLING_EPSILON) return false;
    }
    return true;
  }

  private static getFrustumCornersWorldSpace(invVpMatrix: mat4): vec3[] {
    const ndcCorners: vec4[] = [
      [-1, -1, -1, 1], [-1, 1, -1, 1], [1, 1, -1, 1], [1, -1, -1, 1],
      [-1, -1, 1, 1], [-1, 1, 1, 1], [1, 1, 1, 1], [1, -1, 1, 1],
    ].map(v => vec4.fromValues(v[0], v[1], v[2], v[3]));

    return ndcCorners.map(ndc => {
      const worldVec4 = vec4.transformMat4(vec4.create(), ndc, invVpMatrix);
      if (worldVec4[3] !== 0) vec4.scale(worldVec4, worldVec4, 1.0 / worldVec4[3]);
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
    passEncoder.setBindGroup(0, this.bindGroup);

    // Set the shared buffers ONCE
    passEncoder.setVertexBuffer(0, this.sharedVertexBuffer);
    passEncoder.setIndexBuffer(this.sharedIndexBuffer, INDEX_FORMAT); // Use constant for format

    let totalTriangles = 0;
    let drawnChunks = 0;

    // Draw all visible chunks using offsets
    for (const info of visibleChunkInfos) {
      // Use firstIndex and baseVertex for drawing sub-ranges
      passEncoder.drawIndexed(info.indexCount, 1, info.firstIndex, info.baseVertex, 0);
      totalTriangles += info.indexCount / 3;
      drawnChunks++;
    }

    // Return only drawn count and triangles, culling happened before this call
    return { drawnChunks, totalTriangles };
  }

  // --- Render Frame ---
  public renderFrame(
    cameraPosition: vec3,
    cameraPitch: number,
    cameraYaw: number,
    highlightedBlockPositions: vec3[],
    fov: number,
    debugCamera?: {
      position: vec3,
      target: vec3
    },
    enableDebugView = true
  ): { totalTriangles: number; drawnChunks: number } {

    this.depthTexture = this.configureDepthTexture(this.device, this.depthTexture);
    const aspect = this.canvasWidth/this.canvasHeight;

    // Calculate main camera matrices
    Renderer.updateViewMatrix(this.viewMatrix, cameraPosition, cameraPitch, cameraYaw);
    mat4.perspective(this.projectionMatrix, fov, aspect, 0.1, 1000.0);
    mat4.multiply(this.vpMatrix, this.projectionMatrix, this.viewMatrix);

    // Calculate Sky VP Matrix (Rotation only)
    const skyViewMatrix = mat4.clone(this.viewMatrix);
    skyViewMatrix[12] = 0; // Zero out translation X
    skyViewMatrix[13] = 0; // Zero out translation Y
    skyViewMatrix[14] = 0; // Zero out translation Z
    const skyVpMatrix = mat4.create();
    mat4.multiply(skyVpMatrix, this.projectionMatrix, skyViewMatrix); // Projection * RotationOnlyView


    // --- Handle Debug Camera View ---
    const invVpMatrix = mat4.invert(mat4.create(), this.vpMatrix);
    let worldFrustumCorners: vec3[] = [];
    let activeVpMatrix = this.vpMatrix; // Default to main VP matrix

    if (debugCamera && enableDebugView) {
      mat4.lookAt(
        this.viewMatrixDebug,
        debugCamera.position,
        debugCamera.target,
        vec3.fromValues(0, 1, 0)
      );
      mat4.copy(this.projectionMatrixDebug, this.projectionMatrix);
      mat4.multiply(this.vpMatrixDebug, this.projectionMatrixDebug, this.viewMatrixDebug);
      worldFrustumCorners = Renderer.getFrustumCornersWorldSpace(invVpMatrix);
      activeVpMatrix = this.vpMatrixDebug; // Use debug VP matrix for main scene if active
    }

    // --- Write Uniform Buffers BEFORE Render Pass ---
    // Write main VP matrix (either normal or debug)
    this.device.queue.writeBuffer(this.uniformBuffer, 0, activeVpMatrix as Float32Array);
    // Prepare combined uniform data for the main buffer
    const uniformData = new Float32Array(24); // 16 MVP + 4 LightDir + 4 LightColor + 4 Ambient + pad? = 24 floats
    uniformData.set(activeVpMatrix); // MVP matrix at offset 0 (16 floats)

    // Define light data (hardcoded for now)
    const lightDirection = vec3.normalize(vec3.create(), [0.8, 0.6, 0.2]);
    const lightColor = [1.0, 1.0, 0.5]; // Slightly yellowish sunlight
    const ambientIntensity = 0.7;

    // Set light direction (offset 16 floats = 64 bytes) - vec3 padded to vec4
    uniformData.set(lightDirection, 16);
    // Set light color (offset 16 + 4 = 20 floats = 80 bytes) - vec3 padded to vec4
    uniformData.set(lightColor, 20);
    // Set ambient intensity (offset 20 + 3 = 23 floats? NO -> WGSL offset is 92 bytes / 4 bytes/float = 23)
    uniformData[23] = ambientIntensity; // Offset 23

    // Write the combined data
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniformData);

    // Write Sky VP matrix (always uses rotation-only main camera view)
    this.device.queue.writeBuffer(this.skyUniformBuffer, 0, skyVpMatrix as Float32Array);


    // --- Begin Render Pass ---
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          // Don't clear here if the sky covers the whole screen
          // clearValue: { r: 0.1, g: 0.1, b: 0.15, a: 1.0 },
          loadOp: "clear", // Clear initially
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0, // Important: Clear depth to 1.0
        depthLoadOp: "clear", // Clear depth at the start of the pass
        depthStoreOp: "store", // Keep depth values for scene geometry
      },
    });

    // --- Draw Sky FIRST ---
    passEncoder.setPipeline(this.skyPipeline);
    passEncoder.setBindGroup(0, this.skyBindGroup); // Use sky's bind group/uniforms
    passEncoder.setVertexBuffer(0, this.skyVertexBuffer);
    passEncoder.draw(36); // Draw the 36 vertices of the cube


    // --- Cull Chunks and Prepare Visible List ---
    // (Culling uses the *main* camera's VP matrix, even if debug view is active)
    const frustumPlanes = extractFrustumPlanes(this.vpMatrix);
    const visibleChunkInfos: ChunkGeometryInfo[] = [];
    let culledChunks = 0;
    const allChunkInfos = this.chunkManager.chunkGeometryInfo.values();

    for (const info of allChunkInfos) {
      if (Renderer.intersectFrustumAABB(frustumPlanes, info.aabb)) {
        visibleChunkInfos.push(info);
      } else {
        culledChunks++;
      }
    }

    // --- Draw Voxel Scene ---
    // Voxel scene uses the 'activeVpMatrix' (either normal or debug) via the main uniform buffer
    const sceneStats = this.drawVoxelScene(passEncoder, visibleChunkInfos);

    // Update debug info
    this.debugInfo.totalChunks = this.chunkManager.chunkGeometryInfo.size;
    this.debugInfo.drawnChunks = sceneStats.drawnChunks;
    this.debugInfo.culledChunks = culledChunks;
    this.debugInfo.totalTriangles = sceneStats.totalTriangles;


    // --- Prepare and Draw Highlights ---
    // Highlights also use the 'activeVpMatrix' via the main uniform buffer
    let totalHighlightVertices = 0;
    if (highlightedBlockPositions.length > 0) {
        const numberOfHighlightedCubes = highlightedBlockPositions.length;
        const highlightVertexData = this.generateHighlightVertices(highlightedBlockPositions);
        totalHighlightVertices = numberOfHighlightedCubes * 24;

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

        passEncoder.setPipeline(this.highlightPipeline);
        passEncoder.setBindGroup(0, this.bindGroup); // Use main bind group
        passEncoder.setVertexBuffer(0, this.highlightVertexBuffer);
        passEncoder.draw(totalHighlightVertices, 1, 0, 0);
    }


    // --- Draw Debug Lines ---
    // Debug lines also use the 'activeVpMatrix' via the main uniform buffer
    if (enableDebugView && ENABLE_CHUNK_DEBUG_LINES) {
        // NO need to write uniform buffer here, it was set before the pass

        const lineData = generateDebugLineVertices(
          this.chunkManager,
          frustumPlanes, // Still cull/color based on main frustum
          worldFrustumCorners, // Draw the main frustum corners if debug view active
          cameraPosition // Player position for coloring
        );
        // ... (debug line buffer resize and write remains the same) ...
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
        this.device.queue.writeBuffer(this.debugLineBuffer, 0, lineData);

        // drawDebugLines uses the Renderer instance to get the linePipeline and debugLineBuffer
        // It implicitly uses the currently bound main bind group (binding 0) for the VP matrix
        drawDebugLines(passEncoder, this, lineData);
    }

    // --- Draw Crosshair ---
    // Crosshair uses the UI uniform buffer/bind group (identity matrix)
    const identityMatrix = mat4.create();
    this.device.queue.writeBuffer(this.uiUniformBuffer, 0, identityMatrix as Float32Array);

    passEncoder.setPipeline(this.highlightPipeline); // Use highlight pipe (no depth write)
    passEncoder.setBindGroup(0, this.uiBindGroup);   // Use UI bind group
    passEncoder.setVertexBuffer(0, this.crosshairVertexBuffer);
    passEncoder.draw(this.crosshairVertexCount);


    // --- Finish Up ---
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // Return overall stats
    return {
      totalTriangles: sceneStats.totalTriangles,
      drawnChunks: sceneStats.drawnChunks,
    };
  }

    // Method to resize canvas and internal textures
    public resize(width: number, height: number) {
        if (width <= 0 || height <= 0) return; // Avoid issues with zero/negative size

        const canvas = this.context.canvas as HTMLCanvasElement;
        if (canvas.width === width && canvas.height === height) {
            return; // No change needed
        }

        canvas.width = width;
        canvas.height = height;
        this.canvasWidth = width; // Update internal dimensions first
        this.canvasHeight = height;

        // Reconfigure depth texture
        this.depthTexture = this.configureDepthTexture(this.device, this.depthTexture);

        // Update aspect ratio and crosshair buffer
        this.aspect = this.canvasWidth / this.canvasHeight;
        this.updateCrosshairBuffer(this.aspect);

        console.log(`Renderer resized to ${width}x${height}`);
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

    // Added helper to generate wireframe cube vertices
    private generateHighlightVertices(positions: vec3[]): Float32Array {
        const verticesPerCube = 24; // 12 lines * 2 vertices per line
        const floatsPerVertex = 6; // 3 pos, 3 color
        const data = new Float32Array(positions.length * verticesPerCube * floatsPerVertex);
        let offset = 0;

        const cubeEdges = [
            [0, 0, 0], [1, 0, 0], // Bottom face
            [1, 0, 0], [1, 0, 1],
            [1, 0, 1], [0, 0, 1],
            [0, 0, 1], [0, 0, 0],
            [0, 1, 0], [1, 1, 0], // Top face
            [1, 1, 0], [1, 1, 1],
            [1, 1, 1], [0, 1, 1],
            [0, 1, 1], [0, 1, 0],
            [0, 0, 0], [0, 1, 0], // Connecting edges
            [1, 0, 0], [1, 1, 0],
            [1, 0, 1], [1, 1, 1],
            [0, 0, 1], [0, 1, 1],
        ];

        for (const pos of positions) {
            for (let i = 0; i < cubeEdges.length; i += 2) {
                const start = cubeEdges[i];
                const end = cubeEdges[i+1];

                // Start vertex
                data[offset++] = pos[0] + start[0];
                data[offset++] = pos[1] + start[1];
                data[offset++] = pos[2] + start[2];
                data[offset++] = HIGHLIGHT_COLOR[0];
                data[offset++] = HIGHLIGHT_COLOR[1];
                data[offset++] = HIGHLIGHT_COLOR[2];

                // End vertex
                data[offset++] = pos[0] + end[0];
                data[offset++] = pos[1] + end[1];
                data[offset++] = pos[2] + end[2];
                data[offset++] = HIGHLIGHT_COLOR[0];
                data[offset++] = HIGHLIGHT_COLOR[1];
                data[offset++] = HIGHLIGHT_COLOR[2];
            }
        }
        return data;
    }
}
