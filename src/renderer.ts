import { mat4, vec3, vec4 } from "gl-matrix";
import type { ChunkMesh } from "./chunk";
import { drawDebugLines, generateDebugLineVertices } from "./renderer.debug";
import { extractFrustumPlanes } from "./renderer.util";

// --- Constants ---
const FRUSTUM_CULLING_EPSILON = 1e-5;
export const DEBUG_COLOR_CULLED = [1, 0, 0]; // Red
export const DEBUG_COLOR_DRAWN = [0, 1, 0]; // Green
export const DEBUG_COLOR_FRUSTUM = [0.8, 0.8, 0]; // Yellow
export const DEBUG_COLOR_PLAYER = [0, 0.8, 0.8]; // Cyan
const INITIAL_DEBUG_LINE_BUFFER_SIZE = 1024 * 6 * 4 * 10 * 20; // ~20k lines
const INITIAL_HIGHLIGHT_BUFFER_SIZE = 1024 * 6; // Enough for ~42 cubes initially
const HIGHLIGHT_COLOR = [1.0, 1.0, 0.0]; // Yellow

// @ts-ignore
import voxelShaderCode from "./shaders/voxel.wsgl" with { type: "text" };
// @ts-ignore
import lineShaderCode from "./shaders/line.wsgl" with { type: "text" };
// @ts-ignore
import highlightShaderCode from "./shaders/highlight.wsgl" with { type: "text" };
// @ts-ignore
import cullChunksShader from "./shaders/cullChunks.wsgl" with { type: "text" };

// --- Frustum Culling Types ---
/** Represents a plane equation: Ax + By + Cz + D = 0 */
export type Plane = vec4; // [A, B, C, D]

// --- Renderer Class ---
export class Renderer {
  public device: GPUDevice;
  public context: GPUCanvasContext;
  public presentationFormat: GPUTextureFormat;
  public voxelPipeline: GPURenderPipeline;
  public linePipeline: GPURenderPipeline;
  public highlightPipeline: GPURenderPipeline;
  public uniformBuffer: GPUBuffer;
  public bindGroup: GPUBindGroup;
  public depthTexture: GPUTexture;
  public debugLineBuffer: GPUBuffer;
  public debugLineBufferSize: number;
  public highlightVertexBuffer: GPUBuffer;
  public highlightVertexBufferSize: number;
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
  private canvasWidth = 0;
  private canvasHeight = 0;

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    presentationFormat: GPUTextureFormat,
    voxelPipeline: GPURenderPipeline,
    linePipeline: GPURenderPipeline,
    highlightPipeline: GPURenderPipeline,
    uniformBuffer: GPUBuffer,
    bindGroup: GPUBindGroup,
    debugLineBuffer: GPUBuffer,
    debugLineBufferSize: number,
    highlightVertexBuffer: GPUBuffer,
    highlightVertexBufferSize: number
  ) {
    this.device = device;
    this.context = context;
    this.presentationFormat = presentationFormat;
    this.voxelPipeline = voxelPipeline;
    this.linePipeline = linePipeline;
    this.highlightPipeline = highlightPipeline;
    this.uniformBuffer = uniformBuffer;
    this.bindGroup = bindGroup;

    this.debugLineBuffer = debugLineBuffer;
    this.debugLineBufferSize = debugLineBufferSize;
    this.highlightVertexBuffer = highlightVertexBuffer;
    this.highlightVertexBufferSize = highlightVertexBufferSize;

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
    const voxelPipeline = Renderer.createVoxelPipeline(
      device,
      presentationFormat,
      pipelineLayout
    );
    const linePipeline = Renderer.createLinePipeline(
      device,
      presentationFormat,
      pipelineLayout
    );
    const highlightPipeline = Renderer.createHighlightPipeline(
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

    // --- Highlight Vertex Buffer ---
    const highlightVertexBuffer = device.createBuffer({
      label: "Highlight Vertex Buffer",
      size: INITIAL_HIGHLIGHT_BUFFER_SIZE,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });

    return new Renderer(
      device,
      context,
      presentationFormat,
      voxelPipeline,
      linePipeline,
      highlightPipeline,
      uniformBuffer,
      bindGroup,
      debugLineBuffer,
      INITIAL_DEBUG_LINE_BUFFER_SIZE,
      highlightVertexBuffer,
      INITIAL_HIGHLIGHT_BUFFER_SIZE
    );
  }

  // --- Pipeline Creation (Static because they don't depend on instance state) ---
  private static createVoxelPipeline(
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
        depthCompare: "less", // TEMPORARY: Rule out depth issues
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
    aabb: { min: vec3; max: vec3 }
  ): boolean {
    const { min, max } = aabb;
    for (let i = 0; i < 6; i++) {
      const plane = planes[i];
      const positiveVertex: vec3 = [
        plane[0] > 0 ? max[0] : min[0],
        plane[1] > 0 ? max[1] : min[1],
        plane[2] > 0 ? max[2] : min[2],
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
    chunkMeshes: Map<string, ChunkMesh>,
    cullingFn: (mesh: ChunkMesh) => boolean
  ): { drawnChunks: number; culledChunks: number; totalTriangles: number } {
    passEncoder.setPipeline(this.voxelPipeline);
    passEncoder.setBindGroup(0, this.bindGroup); // Uses current VP matrix in buffer

    let totalTriangles = 0;
    let drawnChunks = 0;
    let culledChunks = 0;

    for (const mesh of chunkMeshes.values()) {
      if (cullingFn(mesh)) {
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

  // --- Render Frame ---
  public renderFrame(
    cameraPosition: vec3,
    cameraPitch: number,
    cameraYaw: number,
    chunkMeshes: Map<string, ChunkMesh>,
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

    // Calculate inverse main VP for frustum corners
    const invVpMatrix = mat4.invert(mat4.create(), this.vpMatrix);
    let worldFrustumCorners: vec3[] = [];

    // Calculate debug camera matrices
    if (debugCamera) {
      mat4.lookAt(
        this.viewMatrixDebug,
        debugCamera.position,
        debugCamera.target,
        vec3.fromValues(0, 1, 0)
      );
      mat4.copy(this.projectionMatrixDebug, this.projectionMatrix);
      mat4.multiply(this.vpMatrixDebug, this.projectionMatrixDebug, this.viewMatrixDebug);
      worldFrustumCorners = Renderer.getFrustumCornersWorldSpace(invVpMatrix);
    }

    // Write the MAIN camera matrix before the render pass for voxel drawing
    this.device.queue.writeBuffer(this.uniformBuffer, 0, this.vpMatrix as Float32Array);

    // --- Begin Render Pass ---
    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.context.getCurrentTexture().createView();
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
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    };
    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

    // --- Draw Voxel Scene ---
    const frustumPlanes = extractFrustumPlanes(this.vpMatrix);
    const sceneStats = this.drawVoxelScene(
      passEncoder,
      chunkMeshes,
      (mesh) => Renderer.intersectFrustumAABB(frustumPlanes, mesh.aabb)
    );
    this.debugInfo.totalChunks = chunkMeshes.size;
    this.debugInfo.drawnChunks = sceneStats.drawnChunks;
    this.debugInfo.culledChunks = sceneStats.culledChunks;
    this.debugInfo.totalTriangles = sceneStats.totalTriangles;

    // --- Prepare Highlight Data ---
    let totalHighlightVertices = 0;
    if (highlightedBlockPositions.length > 0) {
        const numberOfHighlightedCubes = highlightedBlockPositions.length;
        const highlightVertexData = this.generateHighlightVertices(highlightedBlockPositions);
        totalHighlightVertices = numberOfHighlightedCubes * 24; // 24 vertices per cube

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
    }

    // --- Draw Debug Info ---
    let lineData: Float32Array | null = null;
    if (enableDebugView) {
      lineData = generateDebugLineVertices(
        this,
        chunkMeshes,
        frustumPlanes,
        worldFrustumCorners,
        cameraPosition
      );
    }

    if (lineData && lineData.length > 0) {
      // Ensure buffer is large enough (or resize)
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

      drawDebugLines(passEncoder, this, lineData);

      // --- Draw Highlights (using DEBUG camera for now) ---
      if (totalHighlightVertices > 0) {
          passEncoder.setPipeline(this.highlightPipeline);
          passEncoder.setBindGroup(0, this.bindGroup); // Still needs the VP matrix
          passEncoder.setVertexBuffer(0, this.highlightVertexBuffer);
          passEncoder.draw(totalHighlightVertices, 1, 0, 0); // Draw all generated vertices
      }

      // Restore MAIN camera matrix if we were in debug view
    } 
    if (debugCamera) {
      this.device.queue.writeBuffer(this.uniformBuffer, 0, this.vpMatrixDebug as Float32Array);
    } else {
      this.device.queue.writeBuffer(this.uniformBuffer, 0, this.vpMatrix as Float32Array);
    }


    // --- Finish Up ---
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);

    // No need to return updatedDepthTexture as it's internal state now
    return {
      totalTriangles: sceneStats.totalTriangles,
      drawnChunks: sceneStats.drawnChunks,
    };
  }

    // Method to resize canvas and internal textures
    public resize(width: number, height: number) {
        if (width === 0 || height === 0) return; // Avoid issues with zero size

        const canvas = this.context.canvas as HTMLCanvasElement;
        if (canvas.width === width && canvas.height === height) {
            return; // No change needed
        }

        canvas.width = width;
        canvas.height = height;

        // Reconfigure depth texture
        this.depthTexture = this.configureDepthTexture(this.device, this.depthTexture);
        this.canvasWidth = width;
        this.canvasHeight = height;

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
