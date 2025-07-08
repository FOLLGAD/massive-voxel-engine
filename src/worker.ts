/// <reference types="bun-types" />

import { vec3 } from "gl-matrix";
import { Terrain } from "./terrain";
import { Chunk } from "./chunk";
import { ENABLE_GREEDY_MESHING } from "./config";
import log from "./logger";

// Initialize terrain generator
const terrain = new Terrain();

self.onmessage = (event: MessageEvent) => {
  const type = event.data.type;
  const _position = event.data.position;
  
  if (!_position) {
    log.warn("Worker", "No position provided in message");
    return;
  }

  const position = vec3.fromValues(_position[0], _position[1], _position[2]);
  if (type === "requestChunk") {
    try {
      log("Worker", `Generating chunk at position ${position[0]},${position[1]},${position[2]}`);
      
      const chunk = terrain.generateTerrain(position);

      if (chunk.data.length > 0) {
        log("Worker", `Chunk generated, sending data for position ${position[0]},${position[1]},${position[2]}`);
        // Send voxel data for physics and storage (main thread handles storage)
        self.postMessage({
          type: "chunkDataAvailable",
          position,
          voxels: chunk.data,
        });
      } else {
      }

      log("Worker", `Generating mesh for position ${position[0]},${position[1]},${position[2]}`);
      
      const mesh = chunk.generateMesh(); // Calls the appropriate mesher based on the flag
      log("Worker", `Mesh generated: ${mesh.vertices.length} vertices, ${mesh.indices.length} indices`);

      self.postMessage({
        type: "chunkMeshUpdated",
        position,
        vertices: mesh.vertices.buffer,
        indices: mesh.indices.buffer,
        visibilityBits: chunk.generateVisibilityMatrix(),
      });
    } catch (error) {
      console.error("❌ Worker error during processing:", error);
      log.error("Worker", "Error during mesh generation or posting:", error);
    }
  } else if (type === "renderChunk") {
    const { position, data: dataBuffer } = event.data;
    const data = new Uint8Array(dataBuffer);
    const chunk = new Chunk(position, data);
    const mesh = chunk.generateMesh();

    // Send modified chunk data back to main thread for storage
    self.postMessage({
      type: "chunkMeshUpdated",
      position: position,
      vertices: mesh.vertices.buffer,
      indices: mesh.indices.buffer,
      visibilityBits: chunk.generateVisibilityMatrix(),
      modifiedChunkData: chunk.data, // Include modified data for storage
    });
  } else {
    log.warn("Worker", `Unknown message type received: ${type}`);
  }
};

log(
  "Worker",
  `Worker script initialized. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
