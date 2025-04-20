import { Chunk } from "./chunk";
import { CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z } from "./common/constants";
import { VoxelType } from "./common/voxel-types";
import { mkSimplexNoise, mulberry32, type SimplexNoise } from "./noise";

const seedFn = mulberry32(12345);

export class Terrain {
  private noise: SimplexNoise;

  constructor() {
    this.noise = mkSimplexNoise(seedFn);
  }

  generateTerrain(position: { x: number; y: number; z: number }) {
    const chunk = new Chunk(position);
    const stoneDepth = 4; // How deep stone layer goes below dirt/grass
    const terrainScale = 0.001;
    const numOctaves = 4;
    const persistence = 0.5;
    const lacunarity = 2.0;
    const overallAmplitude = 100;
    const baseHeight = 10;

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
            this.noise.noise2D(worldX * frequency, worldZ * frequency) *
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

        const chunkBaseY = chunk.position.y * CHUNK_SIZE_Y;

        // --- Cave generation parameters ---
        const caveScaleX = 0.001; // Smaller scale stretches noise along X
        const caveScaleY = 0.005; // Larger scale makes noise change faster vertically
        const caveScaleZ = 0.001; // Smaller scale stretches noise along Z
        const caveWidth = 0.02; // How close to the median noise value to carve (controls thickness)

        // Secondary noise for cave density/masking
        const caveDensityScale = 0.1; // Larger scale noise for masking regions
        const caveDensityThreshold = 0.4; // Value above which caves are allowed to form (0-1 range)

        for (let y = 0; y < CHUNK_SIZE_Y; y++) {
          const worldY = chunkBaseY + y; // Voxel's world Y position

          // --- Basic Terrain Placement ---
          if (worldY > height) {
            chunk.setVoxel(x, y, z, VoxelType.AIR);
          } else if (worldY === height) {
            chunk.setVoxel(x, y, z, VoxelType.GRASS);
          } else if (worldY > height - stoneDepth) {
            chunk.setVoxel(x, y, z, VoxelType.DIRT);
          } else {
            chunk.setVoxel(x, y, z, VoxelType.STONE);
          }

          // --- Cave Carving (only affects blocks below surface) ---
          if (worldY <= height) {
            // Primary noise for shape
            const primaryCaveNoise =
              (this.noise.noise3D(
                worldX * caveScaleX,
                worldY * caveScaleY,
                worldZ * caveScaleZ
              ) +
                1) /
              2; // Normalize noise to 0-1 range

            // Secondary noise for density/masking
            const densityNoiseValue =
              (this.noise.noise3D(
                // Using slightly offset/different coords for density noise can help
                (worldX + 1000) * caveDensityScale, // Add offset
                worldY * caveDensityScale,
                (worldZ + 1000) * caveDensityScale // Add offset
              ) +
                1) /
              2; // Normalize noise to 0-1 range

            // Check ridge condition AND density condition
            if (
              Math.abs(primaryCaveNoise - 0.5) < caveWidth &&
              densityNoiseValue > caveDensityThreshold
            ) {
              chunk.setVoxel(x, y, z, VoxelType.AIR);
            }
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
