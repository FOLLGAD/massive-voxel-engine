// Uniforms
struct Uniforms {
    mvpMatrix: mat4x4<f32>,
    lightDirection: vec3<f32>,
    lightColor: vec3<f32>,
    ambientIntensity: f32,
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
    let light_direction = normalize(uniforms.lightDirection);
    let ambient_light = uniforms.ambientIntensity;
    let n_len = length(in.normal);
    let surface_normal = select(vec3f(0.0, 0.0, 1.0), normalize(in.normal), n_len > 0.001);
    let diffuse_intensity = max(dot(surface_normal, light_direction), 0.0);
    let brightness = ambient_light + (1.0 - ambient_light) * diffuse_intensity;
    let final_color = in.color * brightness * uniforms.lightColor;
    return vec4<f32>(final_color, 1.0);
}
