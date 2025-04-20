/// <reference lib="webworker" />

import { ENABLE_GREEDY_MESHING } from "./config";
import { Terrain } from "./terrain";

console.log("Worker script loaded.");

const terrain = new Terrain();

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
      const chunk = terrain.generateTerrain(position);
      const mesh = chunk.generateMesh(); // Calls the appropriate mesher based on the flag
      console.log(
        `[Worker] Mesh generated. Vertices count: ${
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
