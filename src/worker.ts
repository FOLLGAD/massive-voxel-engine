/// <reference lib="webworker" />

import {
  CHUNK_SIZE_X,
  CHUNK_SIZE_Y,
  CHUNK_SIZE_Z,
  CHUNK_VOLUME,
} from "./common/constants";
import { VoxelType, isVoxelSolid, getVoxelColor } from "./common/voxel-types";
import { ENABLE_GREEDY_MESHING } from "./config";

console.log("Worker script loaded.");

// Pre-defined vertex data for each face of a cube (centered at 0,0,0) - Needed for Naive Meshing
// Order: position (3 floats)
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

class Chunk {
  public readonly data: Uint8Array;
  public readonly position: { x: number; y: number; z: number };

  constructor(position: { x: number; y: number; z: number }) {
    this.position = position;
    this.data = new Uint8Array(CHUNK_VOLUME);
    this.generateTerrain();
  }

  getVoxel(x: number, y: number, z: number): VoxelType {
    if (
      x < 0 ||
      x >= CHUNK_SIZE_X ||
      y < 0 ||
      y >= CHUNK_SIZE_Y ||
      z < 0 ||
      z >= CHUNK_SIZE_Z
    ) {
      return VoxelType.AIR;
    }
    const index = x + y * CHUNK_SIZE_X + z * CHUNK_SIZE_X * CHUNK_SIZE_Y;
    return this.data[index];
  }

  setVoxel(x: number, y: number, z: number, type: VoxelType): void {
    if (
      x < 0 ||
      x >= CHUNK_SIZE_X ||
      y < 0 ||
      y >= CHUNK_SIZE_Y ||
      z < 0 ||
      z >= CHUNK_SIZE_Z
    ) {
      return;
    }
    const index = x + y * CHUNK_SIZE_X + z * CHUNK_SIZE_X * CHUNK_SIZE_Y;
    this.data[index] = type;
  }

  generateTerrain(): void {
    const baseHeight = CHUNK_SIZE_Y / 2; // Average height level
    const amplitude1 = 10; // Amplitude of the first sine wave
    const frequency1 = 0.03; // Frequency (scale) of the first sine wave
    const amplitude2 = 5; // Amplitude of the second sine wave
    const frequency2 = 0.08; // Frequency of the second sine wave
    const stoneDepth = 5; // How deep stone layer goes below dirt/grass

    // Calculate world offset for noise input
    const worldOffsetX = this.position.x * CHUNK_SIZE_X;
    const worldOffsetZ = this.position.z * CHUNK_SIZE_Z;
    // Note: We don't use worldOffsetY for height calculation, terrain height is independent of chunk Y position

    console.log(
      `Generating procedural terrain for chunk at ${this.position.x},${this.position.y},${this.position.z}`
    );

    for (let x = 0; x < CHUNK_SIZE_X; x++) {
      for (let z = 0; z < CHUNK_SIZE_Z; z++) {
        // Calculate world coordinates for noise input
        const worldX = worldOffsetX + x;
        const worldZ = worldOffsetZ + z;

        // Calculate height using combined sine waves
        const height1 = Math.sin(worldX * frequency1) * amplitude1;
        const height2 = Math.sin(worldZ * frequency2) * amplitude2;
        const height = Math.floor(baseHeight + height1 + height2);

        // Calculate the Y coordinate relative to the chunk's base
        const chunkBaseY = this.position.y * CHUNK_SIZE_Y;

        for (let y = 0; y < CHUNK_SIZE_Y; y++) {
          const worldY = chunkBaseY + y; // Voxel's world Y position

          if (worldY > height) {
            this.setVoxel(x, y, z, VoxelType.AIR);
          } else if (worldY === height) {
            // Top layer is Grass, unless underwater (optional, add later)
            this.setVoxel(x, y, z, VoxelType.GRASS);
          } else if (worldY > height - stoneDepth) {
            // Layer below grass is Dirt
            this.setVoxel(x, y, z, VoxelType.DIRT);
          } else {
            // Deeper layers are Stone
            this.setVoxel(x, y, z, VoxelType.STONE);
          }
        }
      }
    }
    console.log(
      `Terrain generation complete for chunk ${this.position.x},${this.position.y},${this.position.z}`
    );
  }

  // --- Naive Meshing Implementation (with color and normals) ---
  generateNaiveMesh(): { vertices: Float32Array; indices: Uint32Array } {
    console.log("Generating mesh using Naive Meshing (with normals)...");
    const vertices: number[] = []; // Format: [x, y, z, r, g, b, nx, ny, nz, ...]
    const indices: number[] = [];
    let vertexArrayIndex = 0;
    let baseVertexIndexOffset = 0;

    // Calculate world offset for this chunk
    const offsetX = this.position.x * CHUNK_SIZE_X;
    const offsetY = this.position.y * CHUNK_SIZE_Y;
    const offsetZ = this.position.z * CHUNK_SIZE_Z;

    for (let z = 0; z < CHUNK_SIZE_Z; z++) {
      for (let y = 0; y < CHUNK_SIZE_Y; y++) {
        for (let x = 0; x < CHUNK_SIZE_X; x++) {
          const voxelType = this.getVoxel(x, y, z);
          if (!isVoxelSolid(voxelType)) {
            continue; // Skip air blocks
          }

          const color = getVoxelColor(voxelType);

          // Check neighbours
          const neighbors = {
            right: this.getVoxel(x + 1, y, z),
            left: this.getVoxel(x - 1, y, z),
            top: this.getVoxel(x, y + 1, z),
            bottom: this.getVoxel(x, y - 1, z),
            front: this.getVoxel(x, y, z + 1),
            back: this.getVoxel(x, y, z - 1),
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
    console.log(
      `Naive Mesh generation complete. Vertices: ${baseVertexIndexOffset}, Floats: ${vertices.length}, Indices: ${indices.length}`
    );
    return {
      vertices: new Float32Array(vertices),
      indices: new Uint32Array(indices),
    };
  }

  // --- Revised Greedy Meshing Implementation (with color and normals) ---
  generateGreedyMesh(): { vertices: Float32Array; indices: Uint32Array } {
    console.log(
      "Generating mesh using Revised Greedy Meshing (with normals)..."
    );
    const vertices: number[] = []; // Format: [x, y, z, r, g, b, nx, ny, nz, ...]
    const indices: number[] = [];
    let baseVertexIndexOffset = 0;
    const dims = [CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z];

    // Calculate world offset for this chunk
    const offsetX = this.position.x * CHUNK_SIZE_X;
    const offsetY = this.position.y * CHUNK_SIZE_Y;
    const offsetZ = this.position.z * CHUNK_SIZE_Z;

    // Sweep over the 3 dimensions (X, Y, Z)
    for (let d = 0; d < 3; d++) {
      const u = (d + 1) % 3; // Dimension 1 of the slice plane
      const v = (d + 2) % 3; // Dimension 2 of the slice plane

      const x: number[] = [0, 0, 0]; // Current voxel position during sweep
      const mask = new Int8Array(dims[u] * dims[v]); // 2D mask for the slice plane

      // Sweep through slices along dimension 'd'
      for (x[d] = 0; x[d] < dims[d]; ++x[d]) {
        let maskIndex = 0;

        // Generate mask for the current slice plane (u, v)
        for (x[v] = 0; x[v] < dims[v]; ++x[v]) {
          for (x[u] = 0; x[u] < dims[u]; ++x[u]) {
            const type1 = this.getVoxel(x[0], x[1], x[2]);
            const x_neighbor = [...x];
            x_neighbor[d]++;
            const type2 = this.getVoxel(
              x_neighbor[0],
              x_neighbor[1],
              x_neighbor[2]
            );
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
          for (let i = 0; i < dims[u]; ) {
            const maskVal = mask[maskIndex];
            if (maskVal !== 0) {
              const voxelType = Math.abs(maskVal) as VoxelType;
              const isPositiveFace = maskVal > 0;
              const color = getVoxelColor(voxelType);

              // Determine normal based on dimension 'd' and face direction
              const normal: [number, number, number] = [0, 0, 0];
              normal[d] = isPositiveFace ? 1 : -1;

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
              const vertexPosPlane = x[d] + 1;

              // Calculate LOCAL chunk coordinates first
              const v0_local = [...x];
              v0_local[d] = vertexPosPlane; // Corner 0 (i, j)
              const v1_local = [x[0] + du[0], x[1] + du[1], x[2] + du[2]];
              v1_local[d] = vertexPosPlane; // Corner 1 (i+w, j)
              const v2_local = [
                x[0] + du[0] + dv[0],
                x[1] + du[1] + dv[1],
                x[2] + du[2] + dv[2],
              ];
              v2_local[d] = vertexPosPlane; // Corner 2 (i+w, j+h)
              const v3_local = [x[0] + dv[0], x[1] + dv[1], x[2] + dv[2]];
              v3_local[d] = vertexPosPlane; // Corner 3 (i, j+h)

              // Add vertices to the buffer [world_x, world_y, world_z, r, g, b, nx, ny, nz]
              // Apply the world offset here
              vertices.push(v0_local[0] + offsetX, v0_local[1] + offsetY, v0_local[2] + offsetZ, ...color, ...normal);
              vertices.push(v1_local[0] + offsetX, v1_local[1] + offsetY, v1_local[2] + offsetZ, ...color, ...normal);
              vertices.push(v2_local[0] + offsetX, v2_local[1] + offsetY, v2_local[2] + offsetZ, ...color, ...normal);
              vertices.push(v3_local[0] + offsetX, v3_local[1] + offsetY, v3_local[2] + offsetZ, ...color, ...normal);

              // Add indices using standard CCW order relative to the quad vertices v0, v1, v2, v3
              // The GPU's backface culling combined with the view matrix should handle visibility.
              indices.push(
                  baseVertexIndexOffset + 0, baseVertexIndexOffset + 1, baseVertexIndexOffset + 2, // Tri 1: v0-v1-v2
                  baseVertexIndexOffset + 0, baseVertexIndexOffset + 2, baseVertexIndexOffset + 3  // Tri 2: v0-v2-v3
              );

              baseVertexIndexOffset += 4;

              // Zero out maskst
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

    console.log(
      `Revised Greedy Mesh generation complete. Vertices: ${baseVertexIndexOffset}, Floats: ${vertices.length}, Indices: ${indices.length}`
    );
    if (vertices.length === 0 || indices.length === 0) {
      console.warn("Revised Greedy meshing produced no vertices or indices.");
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
} // End Chunk class

// --- Worker Message Handling ---
self.onmessage = (event) => {
  const { type, position } = event.data;
  if (type === "requestChunk") {
    console.log(
      `[Worker] Received requestChunk for ${JSON.stringify(
        position
      )}. Greedy enabled: ${ENABLE_GREEDY_MESHING}`
    ); // Log toggle state
    try {
      const chunk = new Chunk(position);
      const mesh = chunk.generateMesh(); // Calls the appropriate mesher based on the flag
      console.log(
        `[Worker] Mesh generated. Vertices count: ${
          mesh.vertices.length / 9
        }, Indices count: ${mesh.indices.length}`
      ); // Vertices are pos+color+normal (9 floats)

      if (mesh.vertices.length > 0 && mesh.indices.length > 0) {
        console.log("[Worker] Posting chunkMeshAvailable...");
        self.postMessage(
          {
            type: "chunkMeshAvailable",
            position: position,
            vertices: mesh.vertices.buffer,
            indices: mesh.indices.buffer,
          },
          [mesh.vertices.buffer, mesh.indices.buffer]
        );
        console.log("[Worker] Posted chunkMeshAvailable.");
      } else {
        console.warn(
          `[Worker] Skipping postMessage for empty mesh at ${JSON.stringify(
            position
          )}`
        );
        console.log("[Worker] Posting chunkMeshEmpty...");
        self.postMessage({ type: "chunkMeshEmpty", position: position });
        console.log("[Worker] Posted chunkMeshEmpty.");
      }
    } catch (error) {
      console.error("[Worker] Error during mesh generation or posting:", error);
    }
  } else {
    console.warn(`[Worker] Unknown message type received: ${type}`);
  }
};

console.log(
  `Worker script initialized. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
