import { Chunk } from "./chunk";
import {
  CHUNK_SIZE_X,
  CHUNK_SIZE_Y,
  CHUNK_SIZE_Z,
  CHUNK_VOLUME,
} from "./common/constants";
import { VoxelType } from "./common/voxel-types";

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
    const X = Math.floor(x) % 256;
    const Y = Math.floor(y) % 256;
    const X_safe = X < 0 ? X + 256 : X; // Ensure positive for array indexing
    const Y_safe = Y < 0 ? Y + 256 : Y; // Ensure positive for array indexing

    // biome-ignore lint/style/noParameterAssign: <explanation>
    x -= Math.floor(x);
    // biome-ignore lint/style/noParameterAssign: <explanation>
    y -= Math.floor(y);

    const u = this.fade(x);
    const v = this.fade(y);

    const p = this.p; // p is already duplicated to length 512
    // Use safe indices
    const A = p[X_safe] + Y_safe;
    const B = p[(X_safe + 1) % 256] + Y_safe; // Use modulo for wrapping X+1

    // We access p again, need modulo for these lookups too
    // Note: p has length 512, A and B can be > 255
    const hashAA = p[A % 256]; // Modulo A
    const hashAB = p[(A + 1) % 256]; // Modulo A+1
    const hashBA = p[B % 256]; // Modulo B
    const hashBB = p[(B + 1) % 256]; // Modulo B+1

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
const perlinNoise = new PerlinNoise(12345);

export class Terrain {
  private perlinNoise: PerlinNoise;

  constructor() {
    this.perlinNoise = perlinNoise;
  }

  generateTerrain(position: { x: number; y: number; z: number }) {
    const chunk = new Chunk(position);
    const baseHeight = Math.floor(CHUNK_SIZE_Y * 0.4); // Lower average height
    const terrainScale = 0.01; // Lower frequency = larger features
    const numOctaves = 1; // More layers = more detail
    const persistence = 0.5; // Amplitude reduction per octave
    const lacunarity = 2.0; // Frequency increase per octave
    const overallAmplitude = CHUNK_SIZE_Y * 0.4; // Max height variation
    const stoneDepth = 6; // How deep stone layer goes below dirt/grass

    // Calculate world offset for noise input
    const worldOffsetX = chunk.position.x * CHUNK_SIZE_X;
    const worldOffsetZ = chunk.position.z * CHUNK_SIZE_Z;

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
            this.perlinNoise.noise(worldX * frequency, worldZ * frequency) *
            amplitude;
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
        const chunkBaseY = chunk.position.y * CHUNK_SIZE_Y;

        for (let y = 0; y < CHUNK_SIZE_Y; y++) {
          const worldY = chunkBaseY + y; // Voxel's world Y position

          if (worldY > height) {
            chunk.setVoxel(x, y, z, VoxelType.AIR);
          } else if (worldY === height) {
            // Make tops slightly higher than water level grass, lower stone
            if (worldY >= baseHeight - 2) {
              chunk.setVoxel(x, y, z, VoxelType.GRASS);
            } else {
              chunk.setVoxel(x, y, z, VoxelType.STONE); // Mountain peaks are stone
            }
          } else if (worldY > height - stoneDepth) {
            // Layer below grass is Dirt
            chunk.setVoxel(x, y, z, VoxelType.DIRT);
          } else {
            // Deeper layers are Stone
            chunk.setVoxel(x, y, z, VoxelType.STONE);
          }
        }
      }
    }
    console.log(
      `Perlin terrain generation complete for chunk ${chunk.position.x},${chunk.position.y},${chunk.position.z}`
    );

    // Add a 10% chance to spawn a floating stone block (Keep chunk?)
    if (Math.random() < 0.1) {
      const stoneX = Math.floor(Math.random() * CHUNK_SIZE_X);
      const stoneY = Math.floor(Math.random() * CHUNK_SIZE_Y);
      const stoneZ = Math.floor(Math.random() * CHUNK_SIZE_Z);
      chunk.setVoxel(stoneX, stoneY, stoneZ, VoxelType.STONE);
      console.log(
        `[Debug] Added random stone block at ${stoneX},${stoneY},${stoneZ} in chunk ${chunk.position.x},${chunk.position.y},${chunk.position.z}`
      );
    }

    return chunk;
  }
}
