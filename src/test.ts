const setupComputeResources = async (device: GPUDevice) => {
    const computeShaderModule = device.createShaderModule({
        code: `
      @group(0) @binding(0) var<storage, read_write> data: array<f32>;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id: vec3u) {
        let idx = global_id.x;
        // Avoid out-of-bounds access
        if (idx >= arrayLength(&data)) {
          return;
        }
        data[idx] = data[idx];
      }
    `
    });

    const initialComputeData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const computeBufferSize = initialComputeData.byteLength;

    const computeInputOutputBuffer = device.createBuffer({
        size: computeBufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(computeInputOutputBuffer, 0, initialComputeData);

    const computeResultBuffer = device.createBuffer({
        size: computeBufferSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: {
            module: computeShaderModule,
            entryPoint: "main"
        }
    });

    const computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: computeInputOutputBuffer }
        }]
    });

    return { computePipeline, computeBindGroup, computeInputOutputBuffer, computeResultBuffer, initialComputeData, computeBufferSize };
};

const runComputePass = (commandEncoder: GPUCommandEncoder, computeResources: {
    computePipeline: GPUComputePipeline,
    computeBindGroup: GPUBindGroup,
    initialComputeData: Float32Array,
    computeInputOutputBuffer: GPUBuffer,
    computeResultBuffer: GPUBuffer,
    computeBufferSize: number
}) => {
    const { computePipeline, computeBindGroup, initialComputeData, computeInputOutputBuffer, computeResultBuffer, computeBufferSize } = computeResources;
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, computeBindGroup);
    const workgroupCount = Math.ceil(initialComputeData.length / 64);
    computePass.dispatchWorkgroups(workgroupCount);
    computePass.end();

    commandEncoder.copyBufferToBuffer(
        computeInputOutputBuffer,
        0, // Source offset
        computeResultBuffer,
        0, // Destination offset
        computeBufferSize
    );
};

const setupRenderResources = async (device: GPUDevice, format: GPUTextureFormat) => {
    const vertices = new Float32Array([
        0, 0.5,  // top
        -0.5, -0.5, // left
        0.5, -0.5  // right
    ]);

    const indices = new Uint16Array([0, 1, 2, 0]);

    const vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(vertexBuffer, 0, vertices);

    const indexBuffer = device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(indexBuffer, 0, indices);

    const indirectData = new Uint32Array([
        3, 1, 0, 0, 0
    ]);
    const indirectBuffer = device.createBuffer({
        size: indirectData.byteLength,
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(indirectBuffer, 0, indirectData);

    const translationUniformBuffer = device.createBuffer({
        size: 2 * Float32Array.BYTES_PER_ELEMENT, // vec2f for translation (x, y)
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shaderModule = device.createShaderModule({
        code: `
    struct Uniforms {
      translation: vec2f,
    }
    @group(0) @binding(0) var<uniform> uniforms: Uniforms;

    @vertex
    fn vs_main(@location(0) position: vec2f) -> @builtin(position) vec4f {
      return vec4f(position + uniforms.translation, 0.0, 0.5);
    }

    @fragment
    fn fs_main() -> @location(0) vec4f {
      return vec4f(1.0, 0.9, 0.3, 1.0);
    }
  `
    });

    const pipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
            module: shaderModule,
            entryPoint: "vs_main",
            buffers: [{
                arrayStride: 8,
                attributes: [{ format: "float32x2", offset: 0, shaderLocation: 0 }]
            }]
        },
        fragment: {
            module: shaderModule,
            entryPoint: "fs_main",
            targets: [{ format }]
        },
        primitive: { topology: "triangle-list" }
    });

    const renderBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{
            binding: 0,
            resource: { buffer: translationUniformBuffer }
        }]
    });

    return { pipeline, vertexBuffer, indexBuffer, indirectBuffer, translationUniformBuffer, renderBindGroup };
};

const runRenderPass = (commandEncoder: GPUCommandEncoder, context: GPUCanvasContext, renderResources: {
    pipeline: GPURenderPipeline,
    vertexBuffer: GPUBuffer,
    indexBuffer: GPUBuffer,
    indirectBuffer: GPUBuffer,
    renderBindGroup: GPUBindGroup
}) => {
    const { pipeline, vertexBuffer, indexBuffer, indirectBuffer, renderBindGroup } = renderResources;
    const textureView = context.getCurrentTexture().createView();

    const pass = commandEncoder.beginRenderPass({
        colorAttachments: [{
            view: textureView,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1 }
        }]
    });

    pass.setPipeline(pipeline);
    pass.setBindGroup(0, renderBindGroup);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, "uint16");
    pass.drawIndexedIndirect(indirectBuffer, 0);
    pass.end();
};

const f = async () => {
    const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
    const adapter = await navigator.gpu.requestAdapter();
    // biome-ignore lint/style/noNonNullAssertion: <explanation>
    const device = await adapter!.requestDevice();
    // biome-ignore lint/style/noNonNullAssertion: <explanation>
    const context = canvas.getContext("webgpu")!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format });

    const computeResources = await setupComputeResources(device);
    const renderResources = await setupRenderResources(device, format);

    function frame() {
        const commandEncoder = device.createCommandEncoder();

        // Update translation uniform for render pass
        const time = performance.now() / 1000; // time in seconds
        const translation = new Float32Array([Math.sin(time) * 0.3, Math.cos(time) * 0.3]);
        device.queue.writeBuffer(renderResources.translationUniformBuffer, 0, translation);

        runComputePass(commandEncoder, computeResources);
        runRenderPass(commandEncoder, context, renderResources);

        device.queue.submit([commandEncoder.finish()]);

        // if (!frame.hasReadBack) { // Ensure we only do this once
        device.queue.onSubmittedWorkDone().then(async () => {
            await computeResources.computeResultBuffer.mapAsync(GPUMapMode.READ);
            const results = new Float32Array(computeResources.computeResultBuffer.getMappedRange());
            console.log("Compute shader results:", results.slice());
            computeResources.computeResultBuffer.unmap();
        });
        // frame.hasReadBack = true;
        // }

        requestAnimationFrame(frame);
    }
    // frame.hasReadBack = false; // Initialize static property
    requestAnimationFrame(frame);
}

f()