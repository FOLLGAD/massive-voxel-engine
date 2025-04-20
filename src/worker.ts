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

// --- Perlin Noise Implementation ---
// (Based on various sources, simplified for 2D)
class PerlinNoise {
  private p: number[] = []; // Permutation table

  constructor(seed: number = Math.random()) {
    const random = this.seedableRandom(seed);
    this.p = Array.from({ length: 256 }, (_, i) => i);
    // Shuffle p
    for (let i = this.p.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [this.p[i], this.p[j]] = [this.p[j], this.p[i]];
    }
    // Duplicate p to avoid overflow
    this.p = this.p.concat(this.p);
  }

  // Simple seedable pseudo-random number generator
  private seedableRandom(seed: number) {
    let state = seed;
    return () => {
      // Simple LCG (Linear Congruential Generator) parameters
      state = (state * 1103515245 + 12345) % 2147483647;
      return state / 2147483647;
    };
  }

  private fade(t: number): number {
    return t * t * t * (t * (t * 6 - 15) + 10);
  }

  private lerp(t: number, a: number, b: number): number {
    return a + t * (b - a);
  }

  private grad(hash: number, x: number, y: number): number {
    const h = hash & 15; // Use lower 4 bits for gradient direction
    const u = h < 8 ? x : y;
    const v = h < 4 ? y : h === 12 || h === 14 ? x : 0;
    return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
  }

  noise(x: number, y: number): number {
    const X = Math.floor(x) & 255;
    const Y = Math.floor(y) & 255;

    // biome-ignore lint/style/noParameterAssign: <explanation>
    x -= Math.floor(x);
    // biome-ignore lint/style/noParameterAssign: <explanation>
    y -= Math.floor(y);

    const u = this.fade(x);
    const v = this.fade(y);

    const p = this.p;
    const A = p[X] + Y;
    const B = p[X + 1] + Y;

    const hashAA = p[p[A]];
    const hashAB = p[p[A + 1]];
    const hashBA = p[p[B]];
    const hashBB = p[p[B + 1]];

    const gradAA = this.grad(hashAA, x, y);
    const gradAB = this.grad(hashAB, x, y - 1);
    const gradBA = this.grad(hashBA, x - 1, y);
    const gradBB = this.grad(hashBB, x - 1, y - 1);

    const lerpX1 = this.lerp(u, gradBA, gradAA);
    const lerpX2 = this.lerp(u, gradBB, gradAB);
    const result = this.lerp(v, lerpX2, lerpX1);

    // Return value in range [-1, 1] (approximately)
    return result;
  }
}

// --- End Perlin Noise ---

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
    // --- Terrain Generation Parameters ---
    const noiseGen = new PerlinNoise(12345); // Use a fixed seed for consistency
    const baseHeight = Math.floor(CHUNK_SIZE_Y * 0.4); // Lower average height
    const terrainScale = 0.01; // Lower frequency = larger features
    const numOctaves = 5; // More layers = more detail
    const persistence = 0.5; // Amplitude reduction per octave
    const lacunarity = 2.0; // Frequency increase per octave
    const overallAmplitude = CHUNK_SIZE_Y * 0.4; // Max height variation
    const stoneDepth = 6; // How deep stone layer goes below dirt/grass

    // Calculate world offset for noise input
    const worldOffsetX = this.position.x * CHUNK_SIZE_X;
    const worldOffsetZ = this.position.z * CHUNK_SIZE_Z;

    console.log(
      `Generating Perlin terrain for chunk at ${this.position.x},${this.position.y},${this.position.z}`
    );

    for (let x = 0; x < CHUNK_SIZE_X; x++) {
      for (let z = 0; z < CHUNK_SIZE_Z; z++) {
        // Calculate world coordinates for noise input
        const worldX = worldOffsetX + x;
        const worldZ = worldOffsetZ + z;

        // --- Calculate height using multi-octave Perlin noise ---
        let totalNoise = 0;
        let frequency = terrainScale;
        let amplitude = 1.0;
        let maxAmplitude = 0; // Used for normalization

        for (let i = 0; i < numOctaves; i++) {
          totalNoise +=
            noiseGen.noise(worldX * frequency, worldZ * frequency) * amplitude;
          maxAmplitude += amplitude;
          amplitude *= persistence;
          frequency *= lacunarity;
        }

        // Normalize noise to be roughly between -1 and 1, then scale
        const normalizedNoise = totalNoise / maxAmplitude;
        const heightVariation = normalizedNoise * overallAmplitude;
        const height = Math.floor(baseHeight + heightVariation);
        // --- End Height Calculation ---

        // Calculate the Y coordinate relative to the chunk's base
        const chunkBaseY = this.position.y * CHUNK_SIZE_Y;

        for (let y = 0; y < CHUNK_SIZE_Y; y++) {
          const worldY = chunkBaseY + y; // Voxel's world Y position

          if (worldY > height) {
            this.setVoxel(x, y, z, VoxelType.AIR);
          } else if (worldY === height) {
            // Make tops slightly higher than water level grass, lower stone
            if (worldY >= baseHeight - 2) {
              this.setVoxel(x, y, z, VoxelType.GRASS);
            } else {
              this.setVoxel(x, y, z, VoxelType.STONE); // Mountain peaks are stone
            }
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
      `Perlin terrain generation complete for chunk ${this.position.x},${this.position.y},${this.position.z}`
    );

    // Add a 10% chance to spawn a floating stone block (Keep this?)
    if (Math.random() < 0.1) {
      const stoneX = Math.floor(Math.random() * CHUNK_SIZE_X);
      const stoneY = Math.floor(Math.random() * CHUNK_SIZE_Y);
      const stoneZ = Math.floor(Math.random() * CHUNK_SIZE_Z);
      this.setVoxel(stoneX, stoneY, stoneZ, VoxelType.STONE);
      console.log(
        `[Debug] Added random stone block at ${stoneX},${stoneY},${stoneZ} in chunk ${this.position.x},${this.position.y},${this.position.z}`
      );
    }
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
    for (let dimension = 0; dimension < 3; dimension++) {
      const u = (dimension + 1) % 3; // Dimension 1 of the slice plane
      const v = (dimension + 2) % 3; // Dimension 2 of the slice plane

      const x: number[] = [0, 0, 0]; // Current voxel position during sweep
      const mask = new Int8Array(dims[u] * dims[v]); // 2D mask for the slice plane

      // Sweep through slices along dimension 'd'
      for (x[dimension] = -1; x[dimension] < dims[dimension]; ++x[dimension]) {
        let maskIndex = 0;

        // Generate mask for the current slice plane (u, v)
        for (x[v] = 0; x[v] < dims[v]; ++x[v]) {
          for (x[u] = 0; x[u] < dims[u]; ++x[u]) {
            const type1 = this.getVoxel(x[0], x[1], x[2]);
            const x_neighbor = [...x];
            x_neighbor[dimension]++;
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
