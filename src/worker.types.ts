import { vec3 } from "gl-matrix"

export interface WorkerMessageInit {
    type: "init",
    worldName: string,
    worldSeed: number
}

export interface WorkerMessageRequestChunk {
    type: "requestChunk",
    position: vec3
}

export interface WorkerMessageRequestChunkData {
    type: "requestChunkData",
    position: vec3
}

export interface WorkerMessageRenderChunk {
    type: "renderChunk",
    position: vec3,
    data: Uint8Array
}

export interface WorkerMessageDeleteChunk {
    type: "deleteChunk",
    position: vec3
}

export interface WorkerMessageUnloadChunks {
    type: "unloadChunks",
    positions: vec3[]
}

export interface WorkerMessageChunkMeshUpdated {
    type: "chunkMeshUpdated",
    position: vec3,
    data: Uint8Array
}

export type WorkerMessage = WorkerMessageInit | WorkerMessageRequestChunk | WorkerMessageRequestChunkData | WorkerMessageRenderChunk | WorkerMessageDeleteChunk | WorkerMessageUnloadChunks | WorkerMessageChunkMeshUpdated;
