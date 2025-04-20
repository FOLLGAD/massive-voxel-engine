struct ViewParams { // Matches uniform buffer layout
    vp_matrix: mat4x4<f32>,
    // Add frustum planes directly if preferred over extracting in shader
    // plane_left: vec4<f32>, ... plane_far: vec4<f32>
};

struct ChunkMetadata { /* ... as defined above ... */ };
struct DrawIndexedIndirectCommand { /* ... as defined above ... */ };

@group(0) @binding(0) var<uniform> view: ViewParams;
@group(0) @binding(1) var<storage, read> chunk_meta_in: array<ChunkMetadata>;
@group(0) @binding(2) var<storage, read_write> indirect_draw_out: array<DrawIndexedIndirectCommand>;
// Optional counter buffer:
@group(0) @binding(3) var<storage, read_write> visible_count: atomic<u32>;

// Function to extract frustum planes (or pass them in uniform)
// fn extractPlanes(mat: mat4x4<f32>) -> array<vec4<f32>, 6> { ... }

// Function to check AABB intersection (like intersectFrustumAABB)
// fn intersectFrustumAABB(planes: array<vec4<f32>, 6>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool { ... }

const WORKGROUP_SIZE = 64; // Example size, tune based on GPU

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk_index = global_id.x;
    // Prevent out-of-bounds access
    if (chunk_index >= arrayLength(&chunk_meta_in)) {
        return;
    }

    let meta = chunk_meta_in[chunk_index];

    // --- Frustum Culling ---
    // 1. Get frustum planes (either extract from vp_matrix or read from uniform)
    // let planes = extractPlanes(view.vp_matrix); // Or read pre-calculated planes
    // 2. Perform intersection test
    // let is_visible = intersectFrustumAABB(planes, meta.aabb_min, meta.aabb_max);
    let is_visible = true; // Placeholder: Implement actual culling logic here!

    if (is_visible) {
        // Atomically increment counter and get the index for this chunk's command
        let draw_command_index = atomicAdd(&visible_count, 1u);

        // Write the draw command if space available
        if (draw_command_index < arrayLength(&indirect_draw_out)) {
                indirect_draw_out[draw_command_index] = DrawIndexedIndirectCommand(
                meta.index_count, // indexCount
                1u,               // instanceCount
                meta.first_index, // firstIndex
                meta.base_vertex, // baseVertex
                0u                // firstInstance
                );
        }
        // Else: Buffer overflow, too many visible chunks for the allocated buffer size.
    }
}