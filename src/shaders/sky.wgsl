struct Uniforms {
    vpMatrix: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) clipPosition: vec4<f32>,
    @location(0) viewDir: vec3<f32>, // Pass view direction (or world pos)
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32> // Input vertex position (cube vertex)
) -> VertexOutput {
    var out: VertexOutput;
    // Transform cube vertex by rotation-only VP matrix
    out.clipPosition = uniforms.vpMatrix * vec4<f32>(position, 1.0);
    // Ensure sky is drawn at the far plane (z = w)
    out.clipPosition = vec4(out.clipPosition.xy, out.clipPosition.w, out.clipPosition.w);
    // Pass the original vertex position as the direction vector
    // (from origin towards the vertex on the unit cube)
    out.viewDir = position;
    return out;
}

@fragment
fn fs_main(
    @location(0) viewDir: vec3<f32> // Interpolated view direction
) -> @location(0) vec4<f32> {
    let normalizedDir = normalize(viewDir);
    let up = vec3<f32>(0.0, 1.0, 0.0);

    // Simple gradient: lerp between horizon and zenith colors based on Y component
    let horizonColor = vec3<f32>(0.6, 0.8, 1.0); // Light blue/cyan
    let zenithColor = vec3<f32>(0.4, 0.6, 1.0); // Darker blue
    // Use smoothstep for a nicer transition near the horizon
    let t = smoothstep(-0.1, 0.4, normalizedDir.y); // Adjust thresholds as needed

    let finalColor = mix(horizonColor, zenithColor, t);

    return vec4<f32>(finalColor, 1.0);
} 