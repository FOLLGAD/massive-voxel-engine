/// <reference lib="webworker" />

import { ENABLE_GREEDY_MESHING } from "./config";
import { Terrain } from "./terrain";
import log from "./logger";
import { Chunk } from "./chunk";
import { vec3 } from "gl-matrix";

log("Worker", "Worker script loaded.");

const terrain = new Terrain();

// --- Worker Message Handling ---
self.onmessage = (event: MessageEvent) => {
  const { type, position: _position } = event.data as {
    type: string;
    position: vec3;
  };
  const position = vec3.fromValues(_position[0], _position[1], _position[2]);
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
          position,
          voxels: chunk.data,
        });
      }

      const mesh = chunk.generateMesh(); // Calls the appropriate mesher based on the flag
      log(
        "Worker",
        `Mesh generated. Vertices count: ${mesh.vertices.length / 9
        }, Indices count: ${mesh.indices.length}`
      ); // Vertices are pos+color+normal (9 floats)

      self.postMessage({
        type: "chunkMeshUpdated",
        position,
        vertices: mesh.vertices.buffer,
        indices: mesh.indices.buffer,
        visibilityBits: chunk.generateVisibilityMatrix(),
      });
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
      visibilityBits: chunk.generateVisibilityMatrix(),
    });
  } else {
    log.warn("Worker", `Unknown message type received: ${type}`);
  }
};

log(
  "Worker",
  `Worker script initialized. Greedy Meshing Enabled: ${ENABLE_GREEDY_MESHING}`
);
