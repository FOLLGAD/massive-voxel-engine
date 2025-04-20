import { mat4, vec3 } from "gl-matrix";
import type { ChunkMesh } from "./chunk"; // Assuming ChunkMesh interface stays here or moves to common
import { PLAYER_HEIGHT, PLAYER_WIDTH } from "./physics"; // Import player dimensions for hitbox
import log from "./logger";

// --- Renderer State ---
export interface RendererState {
  device: GPUDevice;
  context: GPUCanvasContext;
  presentationFormat: GPUTextureFormat;
  voxelPipeline: GPURenderPipeline;
  uniformBuffer: GPUBuffer;
  bindGroup: GPUBindGroup;
  depthTexture: GPUTexture;
  // Reusable matrices
  viewMatrix: mat4;
  projectionMatrix: mat4;
  vpMatrix: mat4; // View * Projection
}

// --- Shaders ---
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
    @location(2) normal: vec3<f32>,
};

// Vertex shader output structure (interpolated)
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

// Vertex Shader
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = uniforms.mvpMatrix * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.normal = in.normal;
    return out;
}

// Fragment Shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_direction = normalize(vec3<f32>(0.8, 0.6, 0.2));
    let ambient_light = 0.3;
    let n_len = length(in.normal);
    let surface_normal = select(vec3f(0.0, 0.0, 1.0), normalize(in.normal), n_len > 0.001);
    let diffuse_intensity = max(dot(surface_normal, light_direction), 0.0);
    let brightness = ambient_light + (1.0 - ambient_light) * diffuse_intensity;
    let final_color = in.color * brightness;
    return vec4<f32>(final_color, 1.0);
}
`;

const hitboxShaderCode = `
struct Uniforms {
    mvpMatrix: mat4x4<f32>,
};
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return uniforms.mvpMatrix * vec4<f32>(position, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0); // Red color for hitbox
}
`;

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

  // Shaders
  const voxelShaderModule = device.createShaderModule({
    code: voxelShaderCode,
  });

  // Layouts
  const voxelVertexBufferLayout: GPUVertexBufferLayout = {
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

  // Uniform Buffer & Bind Group
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
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
  });

  // Pipelines
  const voxelPipeline = device.createRenderPipeline({
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

  // Depth Texture
  const depthTexture = configureDepthTexture(device, canvas, null);

  // Create matrices once
  const viewMatrix = mat4.create();
  const projectionMatrix = mat4.create();
  const vpMatrix = mat4.create();

  return {
    device,
    context,
    presentationFormat,
    voxelPipeline,
    uniformBuffer,
    bindGroup,
    depthTexture,
    // Add matrices to state
    viewMatrix,
    projectionMatrix,
    vpMatrix,
  };
}

// --- Render Frame ---
export function renderFrame(
  rendererState: RendererState,
  canvas: HTMLCanvasElement, // Needed for aspect ratio and depth texture check
  cameraPosition: vec3, // For camera and hitbox position
  cameraPitch: number,
  cameraYaw: number,
  chunkMeshes: Map<string, ChunkMesh>
): {
  updatedDepthTexture: GPUTexture;
  totalTriangles: number;
} {
  const {
    device,
    context,
    voxelPipeline,
    uniformBuffer,
    bindGroup,
    viewMatrix, // Get from state
    projectionMatrix, // Get from state
    vpMatrix, // Get from state
  } = rendererState;
  let { depthTexture } = rendererState; // depthTexture might be updated

  // Ensure depth texture is correctly sized
  depthTexture = configureDepthTexture(device, canvas, depthTexture);

  // Update matrices (reuse existing ones)
  updateViewMatrix(viewMatrix, cameraPosition, cameraPitch, cameraYaw);
  const aspect = canvas.width / canvas.height;
  mat4.perspective(projectionMatrix, Math.PI / 4, aspect, 0.1, 1000.0);
  mat4.multiply(vpMatrix, projectionMatrix, viewMatrix); // vp = proj * view

  // --- Prepare Render Pass ---
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

  // --- Draw Voxel World ---
  // Write the updated VP matrix to the uniform buffer
  device.queue.writeBuffer(uniformBuffer, 0, vpMatrix as Float32Array);
  passEncoder.setPipeline(voxelPipeline);
  passEncoder.setBindGroup(0, bindGroup);

  let totalTriangles = 0;
  for (const mesh of chunkMeshes.values()) {
    // TODO: Add view frustum culling here - check if mesh.boundingBox intersects frustum

    passEncoder.setVertexBuffer(0, mesh.vertexBuffer);
    passEncoder.setIndexBuffer(mesh.indexBuffer, "uint32");
    passEncoder.drawIndexed(mesh.indexCount);
    totalTriangles += mesh.indexCount / 3;
  }

  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);

  // Return the potentially updated depth texture
  return { updatedDepthTexture: depthTexture, totalTriangles };
}
