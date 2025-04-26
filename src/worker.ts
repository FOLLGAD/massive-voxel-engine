/// <reference lib="webworker" />

import { ENABLE_GREEDY_MESHING } from "./config";
import { Terrain } from "./terrain";
import log from "./logger";
import { Chunk } from "./chunk";

log("Worker", "Worker script loaded.");

const terrain = new Terrain();

// --- Worker Message Handling ---
self.onmessage = (event: MessageEvent) => {
  const { type, position } = event.data;
  if (type === "requestChunk") {
    log(
      "Worker",
      `Received requestChunk for ${JSON.stringify(
        position
      )}. Greedy enabled: ${ENABLE_GREEDY_MESHING}`
    ); // Log toggle state
    try {
      const chunk = terrain.generateTerrain(position);

      if (chunk.data.length > 0) {
        // Send voxel data FIRST (or concurrently) for physics
        self.postMessage({
          type: "chunkDataAvailable",
          position: position,
          voxels: chunk.data,
        });

        const mesh = chunk.generateMesh(); // Calls the appropriate mesher based on the flag
        log(
          "Worker",
          `Mesh generated. Vertices count: ${
            mesh.vertices.length / 9
          }, Indices count: ${mesh.indices.length}`
        ); // Vertices are pos+color+normal (9 floats)

        if (mesh.vertices.length > 0 && mesh.indices.length > 0) {
          self.postMessage({
            type: "chunkMeshUpdated",
            position: position,
            vertices: mesh.vertices.buffer,
            indices: mesh.indices.buffer,
          });
        } else {
          log.warn(
            "Worker",
            `Skipping postMessage for empty mesh at ${JSON.stringify(position)}`
          );
          self.postMessage({ type: "chunkMeshEmpty", position: position });
        }
      }
    } catch (error) {
      log.error("Worker", "Error during mesh generation or posting:", error);
    }
  } else if (type === "renderChunk") {
    const { position, data: dataBuffer } = event.data;
    const data = new Uint8Array(dataBuffer);
    log("Worker", "Rendering chunk", position);
    const chunk = new Chunk(position, data);
    const mesh = chunk.generateMesh();

    self.postMessage({
      type: "chunkMeshUpdated",
      position: position,
      vertices: mesh.vertices.buffer,
      indices: mesh.indices.buffer,
    });
  } else {
    log.warn("Worker", `Unknown message type received: ${type}`);
  }
};

log(
  "Worker",
  `Worker script initialized. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
