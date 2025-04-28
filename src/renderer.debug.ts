import { vec3 } from "gl-matrix";
import {
  type Plane,
  DEBUG_COLOR_DRAWN,
  DEBUG_COLOR_CULLED,
  DEBUG_COLOR_FRUSTUM,
  DEBUG_COLOR_PLAYER,
  Renderer,
} from "./renderer";
import type { ChunkManager } from "./chunk-manager";
import { getPlayerAABB } from "./aabb";

/** Adds line vertices for the 12 edges of a frustum defined by its 8 corners */
export function addFrustumLineVertices(
  vertices: number[],
  corners: vec3[],
  color: number[]
) {
  if (corners.length !== 8) return; // Need exactly 8 corners

  // Define the 12 lines by connecting corner indices (same as AABB)
  // Corner order expected: NBL, NTL, NTR, NBR, FBL, FTL, FTR, FBR
  const lines = [
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    0, // Near face
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    4, // Far face
    0,
    4,
    1,
    5,
    2,
    6,
    3,
    7, // Connecting sides
  ];

  for (let i = 0; i < lines.length; i += 2) {
    const c1 = corners[lines[i]];
    const c2 = corners[lines[i + 1]];
    // Add vertex 1 (pos + color)
    vertices.push(c1[0], c1[1], c1[2], color[0], color[1], color[2]);
    // Add vertex 2 (pos + color)
    vertices.push(c2[0], c2[1], c2[2], color[0], color[1], color[2]);
  }
} /** Generates the Float32Array vertex data for all debug lines */
export function generateDebugLineVertices(
  chunkManager: ChunkManager,
  frustumPlanes: Plane[],
  worldFrustumCorners: vec3[],
  cameraPosition: vec3
): Float32Array {
  const lineVertices: number[] = [];

  // Generate lines for ALL culled/drawn chunks (based on FP camera)
  for (const info of chunkManager.chunkGeometryInfo.values()) {
    const intersects = Renderer.intersectFrustumAABB(frustumPlanes, info.aabb);
    addAABBLineVertices(
      lineVertices,
      info.aabb,
      intersects ? DEBUG_COLOR_DRAWN : DEBUG_COLOR_CULLED
    );
  }

  // Add first-person frustum lines
  addFrustumLineVertices(
    lineVertices,
    worldFrustumCorners,
    DEBUG_COLOR_FRUSTUM
  );

  // Add player hitbox lines
  const playerAABB = getPlayerAABB(cameraPosition);
  addAABBLineVertices(lineVertices, playerAABB, DEBUG_COLOR_PLAYER);

  // Prepare buffer data
  return new Float32Array(lineVertices);
}

/** Adds line vertices (pos[3] + color[3]) for the 12 edges of an AABB */
export function addAABBLineVertices(
  vertices: number[],
  aabb: { min: vec3; max: vec3 },
  color: number[]
) {
  const { min, max } = aabb;
  // Define the 8 corners
  const corners = [
    vec3.fromValues(min[0], min[1], min[2]), // 0: --- Near Bottom Left
    vec3.fromValues(max[0], min[1], min[2]), // 1: +-- Near Bottom Right
    vec3.fromValues(max[0], max[1], min[2]), // 2: ++- Near Top Right
    vec3.fromValues(min[0], max[1], min[2]), // 3: -+- Near Top Left
    vec3.fromValues(min[0], min[1], max[2]), // 4: --+ Far Bottom Left
    vec3.fromValues(max[0], min[1], max[2]), // 5: +-+ Far Bottom Right
    vec3.fromValues(max[0], max[1], max[2]), // 6: +++ Far Top Right
    vec3.fromValues(min[0], max[1], max[2]), // 7: -++ Far Top Left
  ];

  // Define the 12 lines by connecting corner indices
  const lines = [
    0,
    1,
    1,
    2,
    2,
    3,
    3,
    0, // Bottom face
    4,
    5,
    5,
    6,
    6,
    7,
    7,
    4, // Top face
    0,
    4,
    1,
    5,
    2,
    6,
    3,
    7, // Connecting sides
  ];

  for (let i = 0; i < lines.length; i += 2) {
    const c1 = corners[lines[i]];
    const c2 = corners[lines[i + 1]];
    // Add vertex 1 (pos + color)
    vertices.push(c1[0], c1[1], c1[2], color[0], color[1], color[2]);
    // Add vertex 2 (pos + color)
    vertices.push(c2[0], c2[1], c2[2], color[0], color[1], color[2]);
  }
} /** Draws the debug lines using the provided VP matrix */
export function drawDebugLines(
  passEncoder: GPURenderPassEncoder,
  rendererState: Renderer,
  lineData: Float32Array
) {
  // Write the debug view matrix to the uniform buffer
  rendererState.device.queue.writeBuffer(
    rendererState.uniformBuffer,
    0,
    rendererState.vpMatrixDebug as Float32Array
  );
  // Upload the line data to the buffer
  rendererState.device.queue.writeBuffer(
    rendererState.debugLineBuffer,
    0,
    lineData
  );

  // Set pipeline, bind group (which now uses the debug matrix), and buffer
  passEncoder.setPipeline(rendererState.linePipeline);
  passEncoder.setBindGroup(0, rendererState.bindGroup);
  passEncoder.setVertexBuffer(0, rendererState.debugLineBuffer);
  passEncoder.draw(lineData.length / 6);
}
