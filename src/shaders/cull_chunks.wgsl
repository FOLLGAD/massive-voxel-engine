// --- Structs (as defined above) ---
struct ChunkInfo {
    aabb_min: vec3<f32>,
    aabb_max: vec3<f32>,
    id: u32,
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
};

struct Frustum {
    planes: array<vec4<f32>, 6>,
};

struct IndirectDrawCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

// --- Bindings ---
@group(0) @binding(0) var<storage, read> all_chunk_infos: array<ChunkInfo>;
@group(0) @binding(1) var<uniform> frustum_data: Frustum;
@group(0) @binding(2) var<storage, read_write> output_indirect_commands: array<IndirectDrawCommand>;
@group(0) @binding(3) var<storage, read_write> visible_command_count: atomic<u32>; // Counts commands written

// --- Constants ---
const FRUSTUM_CULLING_EPSILON: f32 = 0.00001; // Match your JS constant

// --- Helper: intersectFrustumAABB (Ported from your Renderer.ts) ---
fn intersectFrustumAABB(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    for (var i = 0u; i < 6u; i = i + 1u) {
        let plane = frustum_data.planes[i];

        let positive_vertex = vec3<f32>(
            select(aabb_min.x, aabb_max.x, plane.x > 0.0),
            select(aabb_min.y, aabb_max.y, plane.y > 0.0),
            select(aabb_min.z, aabb_max.z, plane.z > 0.0)
        );

        let distance = dot(plane.xyz, positive_vertex) + plane.w;

        if (distance < -FRUSTUM_CULLING_EPSILON) {
            return false;
        }
    }
    return true;
}

// --- Main Compute Shader ---
@compute @workgroup_size(64) // Example: 64 invocations per workgroup
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let chunk_index = global_id.x;
    let num_total_chunks = arrayLength(&all_chunk_infos);

    if (chunk_index >= num_total_chunks) {
        return; // Out of bounds check
    }

    let current_chunk = all_chunk_infos[chunk_index];

    if (intersectFrustumAABB(current_chunk.aabb_min, current_chunk.aabb_max)) {
        // Chunk is visible, add its draw command to the indirect buffer
        let output_index = atomicAdd(&visible_command_count, 1u); // Get next available slot & increment

        // Ensure output_index is within bounds of output_indirect_commands if it has a fixed size
        // Or, ensure output_indirect_commands is large enough.
        // Your indirectDrawBuffer is already resizable, so this should usually be fine
        // if visible_command_count is reset to 0 each frame before dispatch.

        output_indirect_commands[output_index] = IndirectDrawCommand(
            current_chunk.index_count, // indexCount
            1u,                        // instanceCount
            current_chunk.first_index, // firstIndex
            current_chunk.base_vertex, // baseVertex
            0u                         // firstInstance
        );
    }
}