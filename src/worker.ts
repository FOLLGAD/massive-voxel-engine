/// <reference lib="webworker" />

import { ENABLE_GREEDY_MESHING } from "./config";
import { Terrain } from "./terrain";
import log from "./logger";

log("Worker", "Worker script loaded.");

const terrain = new Terrain();

// --- Worker Message Handling ---
self.onmessage = (event) => {
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

      // Send voxel data FIRST (or concurrently) for physics
      self.postMessage(
        {
          type: "chunkDataAvailable",
          position: position,
          voxels: chunk.data.slice(), // Send the Uint8Array itself
        },
        [chunk.data.buffer.slice()] // Transfer the buffer
      );

      const mesh = chunk.generateMesh(); // Calls the appropriate mesher based on the flag
      log(
        "Worker",
        `Mesh generated. Vertices count: ${
          mesh.vertices.length / 9
        }, Indices count: ${mesh.indices.length}`
      ); // Vertices are pos+color+normal (9 floats)

      if (mesh.vertices.length > 0 && mesh.indices.length > 0) {
        self.postMessage(
          {
            type: "chunkMeshAvailable",
            position: position,
            vertices: mesh.vertices.buffer,
            indices: mesh.indices.buffer,
          },
          [mesh.vertices.buffer, mesh.indices.buffer]
        );
      } else {
        log.warn(
          "Worker",
          `Skipping postMessage for empty mesh at ${JSON.stringify(position)}`
        );
        self.postMessage({ type: "chunkMeshEmpty", position: position });
      }
    } catch (error) {
      log.error("Worker", "Error during mesh generation or posting:", error);
    }
  } else {
    log.warn("Worker", `Unknown message type received: ${type}`);
  }
};

log(
  "Worker",
  `Worker script initialized. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
