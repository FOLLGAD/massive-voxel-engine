import { vec3 } from "gl-matrix";
import { Chunk } from "./chunk";
import { CHUNK_CONFIG } from "./config";
import { VoxelType } from "./common/voxel-types";
import { mkSimplexNoise, mulberry32, type SimplexNoise } from "./noise";

export class Terrain {
  private noise: SimplexNoise;

  constructor(private readonly worldSeed: number) {
    this.noise = mkSimplexNoise(mulberry32(worldSeed));
  }

  epicCaves(worldX: number, worldY: number, worldZ: number) {
    const caveScaleX = 0.01; // Smaller scale stretches noise along X
    const caveScaleY = 0.05; // Larger scale makes noise change faster vertically
    const caveScaleZ = 0.01; // Smaller scale stretches noise along Z

    // Primary noise for shape
    let primaryCaveNoise = this.noise.noise3D(
      worldX * caveScaleX,
      worldY * caveScaleY,
      worldZ * caveScaleZ
    );
    if (primaryCaveNoise < 0) {
      primaryCaveNoise = 0;
    }

    let secondaryCaveNoise = this.noise.noise3D(
      worldX * caveScaleX + -5000,
      worldY * caveScaleY + -5000,
      worldZ * caveScaleZ + -5000
    );
    if (secondaryCaveNoise < 0) {
      secondaryCaveNoise = 0;
    }

    return primaryCaveNoise * secondaryCaveNoise > 0.01;
  }

  bubbleCaves(worldX: number, worldY: number, worldZ: number) {
    let amplitude = 1;
    let frequency = 1;
    let noiseValue = 0;

    const scale = 0.01;
    const threshold = 0.5;
    const octaves = 4;
    const persistence = 0.5;
    const lacunarity = 2.0;

    for (let o = 0; o < octaves; o++) {
      noiseValue +=
        this.noise.noise3D(
          worldX * scale * frequency,
          worldY * scale * frequency,
          worldZ * scale * frequency
        ) * amplitude;

      amplitude *= persistence;
      frequency *= lacunarity;
    }

    return noiseValue > threshold;
  }

  classicCaves(worldX: number, worldY: number, worldZ: number) {
    // --- Cave generation parameters ---
    const caveScaleX = 0.001; // Smaller scale stretches noise along X
    const caveScaleY = 0.001; // Larger scale makes noise change faster vertically
    const caveScaleZ = 0.001; // Smaller scale stretches noise along Z
    const caveWidth = 0.04; // How close to the median noise value to carve (controls thickness)

    // Secondary noise for cave density/masking
    const caveDensityScale = 0.1; // Larger scale noise for masking regions
    const caveDensityThreshold = 0.4; // Value above which caves are allowed to form (0-1 range)
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
    return (
      Math.abs(primaryCaveNoise - 0.5) < caveWidth &&
      densityNoiseValue > caveDensityThreshold
    );
  }

  tunnelCaves(x: number, y: number, z: number) {
    const tunnelScale = 0.005;
    const tunnelScaleY = 0.01;
    const tunnelOffset2 = 2500;
    const tunnelOffset3 = 5000;
    const nx = this.noise.noise3D(
      x * tunnelScale,
      y * tunnelScaleY,
      z * tunnelScale
    );
    const ny = this.noise.noise3D(
      x * tunnelScale + tunnelOffset2,
      y * tunnelScaleY + tunnelOffset2,
      z * tunnelScale + tunnelOffset2
    );
    const nz = this.noise.noise3D(
      2 * x * tunnelScale + tunnelOffset3,
      2 * y * tunnelScaleY + tunnelOffset3,
      2 * z * tunnelScale + tunnelOffset3
    );

    const mag = Math.sqrt(nx * nx + ny * ny + nz * nz);

    // Places where the vector field is "calm" are tunnels
    return mag < 0.2;
  }

  generateTerrain(position: vec3) {
    const terrainConfigs = {
      normal: {
        overallAmplitude: 150,
        baseHeight: 10,
        stoneDepth: 4,
        terrainScale: 0.001,
        numOctaves: 4,
        persistence: 0.5,
        lacunarity: 2.0,
      },
      crazy: {
        overallAmplitude: 500,
        baseHeight: 10,
        stoneDepth: 4,
        terrainScale: 0.001,
        numOctaves: 4,
        persistence: 0.9,
        lacunarity: 2.0,
      },
    };
    const chunk = new Chunk(position);

    const config = terrainConfigs.normal;
    const {
      overallAmplitude,
      baseHeight,
      stoneDepth,
      terrainScale,
      numOctaves,
      persistence,
      lacunarity,
    } = config;

    // Calculate world offset for noise input
    const worldOffsetX = chunk.position[0] * CHUNK_CONFIG.size.x;
    const worldOffsetZ = chunk.position[2] * CHUNK_CONFIG.size.z;

    let isOnlyAir = true;

    for (let x = 0; x < CHUNK_CONFIG.size.x; x++) {
      for (let z = 0; z < CHUNK_CONFIG.size.z; z++) {
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

        const chunkBaseY = chunk.position[1] * CHUNK_CONFIG.size.y;

        for (let y = 0; y < CHUNK_CONFIG.size.y; y++) {
          const worldY = chunkBaseY + y; // Voxel's world Y position

          let voxelType = VoxelType.AIR;

          // --- Basic Terrain Placement ---
          if (worldY > height) {
            voxelType = VoxelType.AIR;
            //if (Math.random() < 3e-6) {
            //  chunk.setVoxel(vec3.fromValues(x, y, z), VoxelType.STAR);
            //} else {
            //}
          } else if (worldY <= height && worldY > height - 3) {
            voxelType = VoxelType.GRASS;
          } else if (worldY > height - stoneDepth) {
            voxelType = VoxelType.DIRT;
          } else {
            voxelType = VoxelType.STONE;
          }

          // --- Cave Carving (only affects blocks below surface) ---
          if (worldY <= height) {
            if (
              this.tunnelCaves(worldX, worldY, worldZ)
              // || this.epicCaves(worldX, worldY, worldZ)
            ) {
              voxelType = VoxelType.AIR;
            }
          }

          if (voxelType !== VoxelType.AIR) {
            isOnlyAir = false;
          }

          chunk.setVoxel(vec3.fromValues(x, y, z), voxelType);
        }
      }
    }

    return chunk;
  }
}
