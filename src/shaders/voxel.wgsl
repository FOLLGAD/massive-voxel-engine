// Uniforms (Data provided by the CPU)
struct Uniforms {
    mvpMatrix: mat4x4<f32>,
};
@binding(0) @group(0) var<uniform> uniforms: Uniforms;

// Vertex shader input structure (matches vertex buffer layout)
struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>, // Add color attribute
};

// Vertex shader output structure (passed to fragment shader)
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>, // Pass color to fragment shader
};

// Vertex Shader
@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    // Transform vertex position by MVP matrix
    out.clip_position = uniforms.mvpMatrix * vec4<f32>(in.position, 1.0);
    out.color = in.color; // Pass color through
    return out;
}

// Fragment Shader
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Use the interpolated vertex color
    return vec4<f32>(in.color, 1.0);
} 