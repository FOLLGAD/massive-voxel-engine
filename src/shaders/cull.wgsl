// cull.wgsl

// Struct to hold chunk data for culling
struct Chunk {
  // AABB - using vec4 for 16-byte alignment
  min: vec4<f32>,
  max: vec4<f32>,

  // Draw command data
  index_count: u32,
  first_index: u32,
  base_vertex: i32,
  // Explicit padding to ensure struct size is a multiple of 16
  _padding: u32,
};

// Struct for the frustum planes
struct Frustum {
  planes: array<vec4<f32>, 6>,
};

// Output draw command for drawIndexedIndirect
struct DrawIndexedIndirectCommand {
  index_count: u32,
  instance_count: u32,
  first_index: u32,
  base_vertex: i32,
  first_instance: u32,
};

@group(0) @binding(0) var<uniform> frustum: Frustum;
@group(0) @binding(1) var<storage, read> chunks: array<Chunk>;
@group(0) @binding(2) var<storage, read_write> indirect_draw_commands: array<DrawIndexedIndirectCommand>;

// Frustum culling check for an AABB
fn intersect_frustum_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
  for (var i = 0u; i < 6u; i = i + 1u) {
    let plane = frustum.planes[i];
    
    // Find the positive vertex of the AABB
    let positive_vertex = vec3<f32>(
      select(aabb_min.x, aabb_max.x, plane.x > 0.0),
      select(aabb_min.y, aabb_max.y, plane.y > 0.0),
      select(aabb_min.z, aabb_max.z, plane.z > 0.0)
    );
    
    // If the positive vertex is outside the plane, the AABB is outside the frustum
    if (dot(plane.xyz, positive_vertex) + plane.w < 0.0) {
      return false;
    }
  }
  return true;
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let chunk_index = global_id.x;
  let num_chunks = arrayLength(&chunks);

  if (chunk_index >= num_chunks) {
    return;
  }

  let chunk = chunks[chunk_index];

  var command : DrawIndexedIndirectCommand;
  command.instance_count = 1u;
  command.first_instance = 0u;

  // Perform frustum culling, using .xyz from the vec4
  if (intersect_frustum_aabb(chunk.min.xyz, chunk.max.xyz)) {
    // If visible, write the real draw command
    command.index_count = chunk.index_count;
    command.first_index = chunk.first_index;
    command.base_vertex = chunk.base_vertex;
  } else {
    // If culled, write a "null" draw command that will be skipped by the GPU
    command.index_count = 0u;
    command.first_index = 0u;
    command.base_vertex = 0;
  }
  indirect_draw_commands[chunk_index] = command;
} 