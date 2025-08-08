/// <reference lib="webworker" />
/// <reference types="bun-types" />

import { vec3 } from "gl-matrix";
import { Terrain } from "./terrain";
import { Chunk } from "./chunk";
import { ENABLE_GREEDY_MESHING } from "./config";
import log from "./logger";
import { VoxelType } from "./common/voxel-types";

// Initialize terrain generator
let terrain: Terrain | null = null;

self.onmessage = async (event: MessageEvent) => {
  const type = event.data.type;

  if (type === "init") {
    const { worldName, worldSeed } = event.data;
    terrain = new Terrain(worldSeed);
  } else if (type === "requestChunk") {
    const { position, data: existingData } = event.data;

    let chunk;
    if (existingData) {
      chunk = new Chunk(position, new Uint8Array(existingData));
    } else {
      chunk = terrain.generateTerrain(position);
      const dataForStorage = chunk.data.buffer.slice(0);
      self.postMessage({
        type: "chunkGenerated",
        position: position,
        voxels: dataForStorage
      }, [dataForStorage as ArrayBuffer]);
    }

    const mesh = chunk.generateMesh();
    self.postMessage({
      type: "chunkMeshUpdated",
      position,
      vertices: mesh.vertices.buffer,
      indices: mesh.indices.buffer,
      visibilityBits: chunk.generateVisibilityMatrix(),
    }, [mesh.vertices.buffer as ArrayBuffer, mesh.indices.buffer as ArrayBuffer]);

  } else if (type === "requestChunkData") {
    const { position, data: existingData } = event.data;
    if (existingData) {
      self.postMessage({
        type: "chunkDataAvailable",
        position,
        voxels: existingData,
      }, [existingData as ArrayBuffer]);
    } else {
      const chunk = terrain.generateTerrain(position);
      const dataForStorage = chunk.data.buffer.slice(0);
      const dataForCache = chunk.data.buffer.slice(0);
      self.postMessage({
        type: "chunkGenerated",
        position: position,
        voxels: dataForStorage
      }, [dataForStorage as ArrayBuffer]);
      self.postMessage({
        type: "chunkDataAvailable",
        position,
        voxels: dataForCache,
      }, [dataForCache as ArrayBuffer]);
    }

  } else if (type === "renderChunk") {
    const { position, data } = event.data;
    const chunkData = new Uint8Array(data);

    const chunk = new Chunk(position, chunkData);

    if (chunk.data.every(voxel => voxel === VoxelType.AIR)) {
      return;
    }

    const mesh = chunk.generateMesh();

    self.postMessage({
      type: "chunkMeshUpdated",
      position: position,
      vertices: mesh.vertices.buffer,
      indices: mesh.indices.buffer,
      visibilityBits: chunk.generateVisibilityMatrix(),
    }, [mesh.vertices.buffer as ArrayBuffer, mesh.indices.buffer as ArrayBuffer]);

  } else if (type === "deleteChunk") {
    self.postMessage({
      type: 'chunkNeedsDeletion',
      position: event.data.position
    });

  } else if (type === "unloadChunks") {
    const { allChunkKeys, playerPosition, loadRadiusXZ, loadRadiusY, unloadBufferXZ, unloadBufferY } = event.data;
    const chunksToUnload: string[] = [];

    const unloadRadiusXZ = loadRadiusXZ + unloadBufferXZ;
    const unloadRadiusY = loadRadiusY + unloadBufferY;

    for (const chunkKey of allChunkKeys) {
      const [x, y, z] = chunkKey.split(',').map(Number);
      const chunkPos = vec3.fromValues(x, y, z);

      const dx = Math.abs(chunkPos[0] - playerPosition[0]);
      const dy = Math.abs(chunkPos[1] - playerPosition[1]);
      const dz = Math.abs(chunkPos[2] - playerPosition[2]);

      if (dx > unloadRadiusXZ || dy > unloadRadiusY || dz > unloadRadiusXZ) {
        chunksToUnload.push(chunkKey);
      }
    }

    if (chunksToUnload.length > 0) {
      self.postMessage({
        type: "chunksToUnload",
        chunks: chunksToUnload,
      });
    }

  } else {
    log.warn("Worker", `Unknown message type received: ${type}`);
    console.error("Unknown message type received", type);
  }
};

log(
  "Worker",
  `Worker script initialized. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
