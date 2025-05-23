import { vec3 } from "gl-matrix";
import {
  CHUNK_SIZE_X,
  CHUNK_SIZE_Y,
  CHUNK_SIZE_Z,
  CHUNK_VOLUME,
} from "./config";
import { VoxelType } from "./common/voxel-types";
import { getVoxelColor, isVoxelSolid } from "./common/voxel-types";
import { ENABLE_GREEDY_MESHING } from "./config";
import log from "./logger";
import type { AABB } from "./aabb";

export const getLocalPosition = (position: vec3) => {
  const x = ((position[0] % CHUNK_SIZE_X) + CHUNK_SIZE_X) % CHUNK_SIZE_X;
  const y = ((position[1] % CHUNK_SIZE_Y) + CHUNK_SIZE_Y) % CHUNK_SIZE_Y;
  const z = ((position[2] % CHUNK_SIZE_Z) + CHUNK_SIZE_Z) % CHUNK_SIZE_Z;
  return vec3.fromValues(x, y, z);
};

export const getChunkOfPosition = (position: vec3) => {
  return vec3.fromValues(
    Math.floor(position[0] / CHUNK_SIZE_X),
    Math.floor(position[1] / CHUNK_SIZE_Y),
    Math.floor(position[2] / CHUNK_SIZE_Z)
  );
};

const CUBE_FACES = {
  // +X (Right)
  right: [
    0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5,
    0.5, 0.5, -0.5, 0.5,
  ],
  // -X (Left)
  left: [
    -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5,
    0.5, -0.5, -0.5, -0.5, -0.5,
  ],
  // +Y (Top)
  top: [
    -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5,
    0.5, 0.5, 0.5, -0.5,
  ],
  // -Y (Bottom)
  bottom: [
    -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5,
    -0.5, -0.5, 0.5, -0.5, 0.5,
  ],
  // +Z (Front)
  front: [
    -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5,
    0.5, -0.5, 0.5, 0.5,
  ],
  // -Z (Back)
  back: [
    0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
    0.5, -0.5, 0.5, 0.5, -0.5,
  ],
};

// Define face normals (used by both meshers)
const FACE_NORMALS: { [key: string]: [number, number, number] } = {
  right: [1, 0, 0],
  left: [-1, 0, 0],
  top: [0, 1, 0],
  bottom: [0, -1, 0],
  front: [0, 0, 1],
  back: [0, 0, -1],
};

const getVoxelIndex = (localPos: vec3): number => {
  return (
    localPos[0] +
    localPos[1] * CHUNK_SIZE_X +
    localPos[2] * CHUNK_SIZE_X * CHUNK_SIZE_Y
  );
};

// --- START Face Connectivity Helpers ---

// Face indices
const FACE_X_PLUS = 0;
const FACE_X_MINUS = 1;
const FACE_Y_PLUS = 2;
const FACE_Y_MINUS = 3;
const FACE_Z_PLUS = 4;
const FACE_Z_MINUS = 5;
const NUM_FACES = 6;

// Helper to check if a local position is on a specific face
const isOnFace = (pos: vec3, faceIndex: number): boolean => {
  switch (faceIndex) {
    case FACE_X_PLUS:
      return pos[0] === CHUNK_SIZE_X - 1;
    case FACE_X_MINUS:
      return pos[0] === 0;
    case FACE_Y_PLUS:
      return pos[1] === CHUNK_SIZE_Y - 1;
    case FACE_Y_MINUS:
      return pos[1] === 0;
    case FACE_Z_PLUS:
      return pos[2] === CHUNK_SIZE_Z - 1;
    case FACE_Z_MINUS:
      return pos[2] === 0;
    default:
      return false;
  }
};

// Helper to get neighbors (direction 0..5 -> +x, -x, +y, -y, +z, -z)
const getNeighbor = (pos: vec3, direction: number): vec3 => {
  const neighbor = vec3.clone(pos);
  const delta: [number, number, number][] = [
    [1, 0, 0], // +X
    [-1, 0, 0], // -X
    [0, 1, 0], // +Y
    [0, -1, 0], // -Y
    [0, 0, 1], // +Z
    [0, 0, -1], // -Z
  ];
  vec3.add(neighbor, neighbor, delta[direction]);
  return neighbor;
};

// Flood fill function to find connected faces for an air component
const findConnectedFaces = (
  startPos: vec3,
  visitedArr: boolean[],
  chunkData: Uint8Array
): number => {
  const startIndex = getVoxelIndex(startPos);
  // Initial check redundant if called correctly, but safe
  if (visitedArr[startIndex] || isVoxelSolid(chunkData[startIndex])) {
    return 0;
  }

  let componentFacesMask = 0;
  const queue: vec3[] = [];
  queue.push(startPos);
  visitedArr[startIndex] = true;

  let head = 0;
  while (head < queue.length) {
    // Using queue as FIFO, check size limit if necessary
    if (queue.length > CHUNK_VOLUME * 2) {
      log.error("Chunk", "Flood fill queue grew too large, aborting.");
      // Return what we found so far, might be incomplete but avoids infinite loops
      return componentFacesMask;
    }

    const currentPos = queue[head++]; // Dequeue

    // Check which boundary faces this voxel touches
    for (let face = 0; face < NUM_FACES; face++) {
      if (isOnFace(currentPos, face)) {
        componentFacesMask |= 1 << face;
      }
    }

    // Add valid, unvisited, air neighbors to queue
    for (let dir = 0; dir < 6; dir++) {
      const neighborPos = getNeighbor(currentPos, dir);

      // Check bounds
      if (
        neighborPos[0] < 0 ||
        neighborPos[0] >= CHUNK_SIZE_X ||
        neighborPos[1] < 0 ||
        neighborPos[1] >= CHUNK_SIZE_Y ||
        neighborPos[2] < 0 ||
        neighborPos[2] >= CHUNK_SIZE_Z
      ) {
        continue;
      }

      const neighborIndex = getVoxelIndex(neighborPos);
      if (
        !visitedArr[neighborIndex] &&
        !isVoxelSolid(chunkData[neighborIndex])
      ) {
        visitedArr[neighborIndex] = true;
        queue.push(neighborPos);
      }
    }
  }

  return componentFacesMask;
};

// Function to get the bit index (0-14) for a pair of faces (face1 < face2)
export const getPairBitIndex = (face1: number, face2: number): number => {
  if (face1 > face2) {
    // biome-ignore lint/style/noParameterAssign: <explanation>
    [face1, face2] = [face2, face1];
  }

  let sumBeforeRow = 0;
  for (let k = 0; k < face1; k++) {
    // Number of pairs starting with k is (NUM_FACES - 1 - k)
    sumBeforeRow += NUM_FACES - 1 - k;
  }
  const indexInRow = face2 - face1 - 1;
  return sumBeforeRow + indexInRow;
};

// --- END Face Connectivity Helpers ---

export class Chunk {
  position: vec3;
  data: Uint8Array;

  constructor(position: vec3, data?: Uint8Array) {
    this.position = position;
    this.data = data ?? new Uint8Array(new SharedArrayBuffer(CHUNK_VOLUME));
  }

  static withData(position: vec3, data: Uint8Array): Chunk {
    const chunk = new Chunk(position, data);
    return chunk;
  }

  getVoxel(localPos: vec3): VoxelType {
    const index = getVoxelIndex(localPos);
    if (
      localPos[0] < 0 ||
      localPos[0] >= CHUNK_SIZE_X ||
      localPos[1] < 0 ||
      localPos[1] >= CHUNK_SIZE_Y ||
      localPos[2] < 0 ||
      localPos[2] >= CHUNK_SIZE_Z
    ) {
      log.warn(
        "Chunk",
        `Calculated index ${index} out of bounds for chunk ${getChunkKey(
          this.position
        )} (size ${CHUNK_VOLUME})`
      );
      return VoxelType.AIR;
    }
    // biome-ignore lint/style/noNonNullAssertion: <explanation>
    return this.data![index];
  }

  setVoxel(localPos: vec3, type: VoxelType): void {
    if (
      localPos[0] < 0 ||
      localPos[0] >= CHUNK_SIZE_X ||
      localPos[1] < 0 ||
      localPos[1] >= CHUNK_SIZE_Y ||
      localPos[2] < 0 ||
      localPos[2] >= CHUNK_SIZE_Z
    ) {
      return;
    }
    const index = getVoxelIndex(localPos);
    // biome-ignore lint/style/noNonNullAssertion: <explanation>
    this.data![index] = type;
  }

  // --- Naive Meshing Implementation (with color and normals) ---
  generateNaiveMesh(): { vertices: Float32Array; indices: Uint32Array } {
    log("Chunk", "Generating mesh using Naive Meshing (with normals)...");
    const vertices: number[] = []; // Format: [x, y, z, r, g, b, nx, ny, nz, ...]
    const indices: number[] = [];
    let vertexArrayIndex = 0;
    let baseVertexIndexOffset = 0;

    // Calculate world offset for this chunk
    const offsetX = this.position[0] * CHUNK_SIZE_X;
    const offsetY = this.position[1] * CHUNK_SIZE_Y;
    const offsetZ = this.position[2] * CHUNK_SIZE_Z;

    for (let z = 0; z < CHUNK_SIZE_Z; z++) {
      for (let y = 0; y < CHUNK_SIZE_Y; y++) {
        for (let x = 0; x < CHUNK_SIZE_X; x++) {
          const localPos = vec3.fromValues(x, y, z);
          const voxelType = this.getVoxel(localPos);
          if (!isVoxelSolid(voxelType)) {
            continue; // Skip air blocks
          }

          const color = getVoxelColor(voxelType);

          // Check neighbours
          const neighbors = {
            right: this.getVoxel(vec3.fromValues(x + 1, y, z)),
            left: this.getVoxel(vec3.fromValues(x - 1, y, z)),
            top: this.getVoxel(vec3.fromValues(x, y + 1, z)),
            bottom: this.getVoxel(vec3.fromValues(x, y - 1, z)),
            front: this.getVoxel(vec3.fromValues(x, y, z + 1)),
            back: this.getVoxel(vec3.fromValues(x, y, z - 1)),
          };

          // Add faces if neighbor is air/transparent
          for (const face of Object.keys(neighbors) as Array<
            keyof typeof neighbors
          >) {
            if (!isVoxelSolid(neighbors[face])) {
              const faceVertices = CUBE_FACES[face]; // Position data for the face
              const normal = FACE_NORMALS[face]; // Get the normal for this face

              // Each face has 6 vertices (2 triangles)
              for (let i = 0; i < 6; i++) {
                const px = faceVertices[i * 3 + 0];
                const py = faceVertices[i * 3 + 1];
                const pz = faceVertices[i * 3 + 2];
                vertices[vertexArrayIndex++] = px + x + offsetX;
                vertices[vertexArrayIndex++] = py + y + offsetY;
                vertices[vertexArrayIndex++] = pz + z + offsetZ; // Position
                vertices[vertexArrayIndex++] = color[0];
                vertices[vertexArrayIndex++] = color[1];
                vertices[vertexArrayIndex++] = color[2]; // Color
                vertices[vertexArrayIndex++] = normal[0];
                vertices[vertexArrayIndex++] = normal[1];
                vertices[vertexArrayIndex++] = normal[2]; // Normal
              }
              indices.push(
                baseVertexIndexOffset + 0,
                baseVertexIndexOffset + 1,
                baseVertexIndexOffset + 2,
                baseVertexIndexOffset + 3,
                baseVertexIndexOffset + 4,
                baseVertexIndexOffset + 5
              );
              baseVertexIndexOffset += 6;
            }
          }
        }
      }
    }
    log(
      "Chunk",
      `Naive Mesh generation complete. Vertices: ${baseVertexIndexOffset}, Floats: ${vertices.length}, Indices: ${indices.length}`
    );
    return {
      vertices: new Float32Array(vertices),
      indices: new Uint32Array(indices),
    };
  }

  // --- Revised Greedy Meshing Implementation (with color and normals) ---
  generateGreedyMesh(): { vertices: Float32Array; indices: Uint32Array } {
    const vertices: number[] = []; // Format: [x, y, z, r, g, b, nx, ny, nz, ...]
    const indices: number[] = [];
    let baseVertexIndexOffset = 0;
    const dims = [CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z];

    // Calculate world offset for this chunk
    const offsetX = this.position[0] * CHUNK_SIZE_X;
    const offsetY = this.position[1] * CHUNK_SIZE_Y;
    const offsetZ = this.position[2] * CHUNK_SIZE_Z;

    // Sweep over the 3 dimensions (X, Y, Z)
    for (let dimension = 0; dimension < 3; dimension++) {
      const u = (dimension + 1) % 3; // Dimension 1 of the slice plane
      const v = (dimension + 2) % 3; // Dimension 2 of the slice plane

      const x = vec3.fromValues(0, 0, 0); // Current voxel position during sweep
      const mask = new Int8Array(dims[u] * dims[v]); // 2D mask for the slice plane

      // Sweep through slices along dimension 'd'
      for (x[dimension] = -1; x[dimension] < dims[dimension]; ++x[dimension]) {
        let maskIndex = 0;

        // Generate mask for the current slice plane (u, v)
        for (x[v] = 0; x[v] < dims[v]; ++x[v]) {
          for (x[u] = 0; x[u] < dims[u]; ++x[u]) {
            const type1 = this.getVoxel(x);
            const x_neighbor = vec3.clone(x);
            x_neighbor[dimension]++;
            const type2 = this.getVoxel(x_neighbor);
            const solid1 = isVoxelSolid(type1);
            const solid2 = isVoxelSolid(type2);

            if (solid1 === solid2) {
              mask[maskIndex++] = 0;
            } else if (solid1) {
              mask[maskIndex++] = type1;
            } else {
              mask[maskIndex++] = -type2;
            }
          }
        }

        // Now, generate geometry from the mask
        maskIndex = 0;
        for (let j = 0; j < dims[v]; ++j) {
          for (let i = 0; i < dims[u];) {
            const maskVal = mask[maskIndex];
            if (maskVal !== 0) {
              const voxelType = Math.abs(maskVal) as VoxelType;
              const isPositiveFace = maskVal > 0;
              const color = getVoxelColor(voxelType);

              // Determine normal based on dimension 'd' and face direction
              const normal: [number, number, number] = [0, 0, 0];
              normal[dimension] = isPositiveFace ? 1 : -1;

              // Calculate width (w) along u dimension
              let w = 1;
              while (i + w < dims[u] && mask[maskIndex + w] === maskVal) {
                w++;
              }

              // Calculate height (h) along v dimension
              let h = 1;
              let done = false;
              while (j + h < dims[v]) {
                for (let k = 0; k < w; ++k) {
                  if (mask[maskIndex + k + h * dims[u]] !== maskVal) {
                    done = true;
                    break;
                  }
                }
                if (done) break;
                h++;
              }

              // --- Add quad ---
              x[u] = i;
              x[v] = j;
              const du = [0, 0, 0];
              du[u] = w;
              const dv = [0, 0, 0];
              dv[v] = h;
              const vertexPosPlane = x[dimension] + 1; // Corrected position

              // Calculate LOCAL chunk coordinates first
              const v0_local = [...x];
              v0_local[dimension] = vertexPosPlane; // Corner 0 (i, j)
              const v1_local = [x[0] + du[0], x[1] + du[1], x[2] + du[2]];
              v1_local[dimension] = vertexPosPlane; // Corner 1 (i+w, j)
              const v2_local = [
                x[0] + du[0] + dv[0],
                x[1] + du[1] + dv[1],
                x[2] + du[2] + dv[2],
              ];
              v2_local[dimension] = vertexPosPlane; // Corner 2 (i+w, j+h)
              const v3_local = [x[0] + dv[0], x[1] + dv[1], x[2] + dv[2]];
              v3_local[dimension] = vertexPosPlane; // Corner 3 (i, j+h)

              // Add vertices to the buffer [world_x, world_y, world_z, r, g, b, nx, ny, nz]
              // Apply the world offset here
              vertices.push(
                v0_local[0] + offsetX,
                v0_local[1] + offsetY,
                v0_local[2] + offsetZ,
                ...color,
                ...normal
              );
              vertices.push(
                v1_local[0] + offsetX,
                v1_local[1] + offsetY,
                v1_local[2] + offsetZ,
                ...color,
                ...normal
              );
              vertices.push(
                v2_local[0] + offsetX,
                v2_local[1] + offsetY,
                v2_local[2] + offsetZ,
                ...color,
                ...normal
              );
              vertices.push(
                v3_local[0] + offsetX,
                v3_local[1] + offsetY,
                v3_local[2] + offsetZ,
                ...color,
                ...normal
              );

              // Add indices. Check if winding order needs reversal.
              if (!isPositiveFace) {
                // Reversed winding order: (0,2,1), (0,3,2)
                indices.push(
                  baseVertexIndexOffset + 0,
                  baseVertexIndexOffset + 2,
                  baseVertexIndexOffset + 1,
                  baseVertexIndexOffset + 0,
                  baseVertexIndexOffset + 3,
                  baseVertexIndexOffset + 2
                );
              } else {
                // Standard CCW winding order: (0,1,2), (0,2,3)
                indices.push(
                  baseVertexIndexOffset + 0,
                  baseVertexIndexOffset + 1,
                  baseVertexIndexOffset + 2,
                  baseVertexIndexOffset + 0,
                  baseVertexIndexOffset + 2,
                  baseVertexIndexOffset + 3
                );
              }
              baseVertexIndexOffset += 4;

              // Zero out mask
              for (let l = 0; l < h; ++l) {
                for (let k = 0; k < w; ++k) {
                  mask[maskIndex + k + l * dims[u]] = 0;
                }
              }
              i += w;
              maskIndex += w;
              continue; // Skip the increment at the end
            }
            i++;
            maskIndex++;
          }
        }
      }
    }

    log(
      "Chunk",
      `Revised Greedy Mesh generation complete. Vertices: ${baseVertexIndexOffset}, Floats: ${vertices.length}, Indices: ${indices.length}`
    );
    if (vertices.length === 0 || indices.length === 0) {
      log.warn(
        "Chunk",
        "Revised Greedy meshing produced no vertices or indices."
      );
      return { vertices: new Float32Array(0), indices: new Uint32Array(0) };
    }
    return {
      vertices: new Float32Array(vertices),
      indices: new Uint32Array(indices),
    };
  }

  // Main generateMesh method chooses which algorithm to use
  generateMesh(): { vertices: Float32Array; indices: Uint32Array } {
    if (ENABLE_GREEDY_MESHING) {
      return this.generateGreedyMesh();
      // biome-ignore lint/style/noUselessElse: <explanation>
    } else {
      return this.generateNaiveMesh();
    }
  }

  // Calculates a 15-bit integer where each bit indicates if two distinct faces
  // are connected by a path of air voxels within the chunk.
  generateVisibilityMatrix(): number {
    const visitedArr = new Array(CHUNK_VOLUME).fill(false);
    let conjoinedSides = 0; // 15-bit result

    for (let z = 0; z < CHUNK_SIZE_Z; z++) {
      for (let y = 0; y < CHUNK_SIZE_Y; y++) {
        for (let x = 0; x < CHUNK_SIZE_X; x++) {
          const pos = vec3.fromValues(x, y, z);
          const index = getVoxelIndex(pos);

          // Start flood fill from any unvisited air voxel
          if (!visitedArr[index] && !isVoxelSolid(this.data[index])) {
            const componentFacesMask = findConnectedFaces(
              pos,
              visitedArr,
              this.data
            );

            // Update conjoinedSides based on face pairs touched by this component
            for (let i = 0; i < NUM_FACES; i++) {
              if ((componentFacesMask & (1 << i)) !== 0) {
                // If face 'i' is touched...
                for (let j = i + 1; j < NUM_FACES; j++) {
                  if ((componentFacesMask & (1 << j)) !== 0) {
                    // ...and face 'j' is also touched, they are connected.
                    const bitIndex = getPairBitIndex(i, j);
                    conjoinedSides |= 1 << bitIndex;
                  }
                }
              }
            }
          }
        }
      }
    }

    return conjoinedSides;
  }
}

export interface ChunkMesh {
  position: vec3;
  vertexBuffer: GPUBuffer;
  indexBuffer: GPUBuffer;
  indexCount: number;
  aabb: AABB; // Add AABB for frustum culling
}

export function getChunkKey(pos: vec3): string {
  return `${pos[0]},${pos[1]},${pos[2]}`;
}
