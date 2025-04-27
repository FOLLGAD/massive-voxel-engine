import type { vec3 } from "gl-matrix";
import { getChunkKey } from "./chunk";

// Define constants needed for calculations
const VERTEX_STRIDE_BYTES = 9 * Float32Array.BYTES_PER_ELEMENT; // Matches renderer's voxelVertexBufferLayout
const INDEX_SIZE_BYTES = 4; // Based on uint32 index format used in renderer

export interface ChunkGeometryInfo {
  /** How many indices to draw for this chunk. */
  indexCount: number;

  /** The byte offset where this chunk's vertex data begins in the sharedVertexBuffer. */
  vertexOffsetBytes: number;

  /** The total size in bytes of this chunk's vertex data block. */
  vertexSizeBytes: number;

  /** The byte offset where this chunk's index data begins in the sharedIndexBuffer. */
  indexOffsetBytes: number;

  /** The total size in bytes of this chunk's index data block. */
  indexSizeBytes: number;

  /** The raw vertex data for this chunk. */
  vertexData: Float32Array;

  /** The raw index data for this chunk. */
  indexData: Uint32Array;

  /** The AABB of the chunk. */
  aabb: { min: vec3; max: vec3 };

  /** The position of the chunk. */
  position: vec3;

  /** The offset (in indices) into the shared index buffer for the first index. */
  firstIndex: number;

  /** The value added to each index before reading from the vertex buffer. */
  baseVertex: number;
}

export interface MemorySpaceInfo {
  offset: number;
  size: number;
  key: string;
}

export class ChunkManager {
  chunkGeometryInfo: Map<string, ChunkGeometryInfo>;

  // Make buffers public for access in debug renderer (simplest approach)
  public device: GPUDevice;
  public sharedVertexBuffer: GPUBuffer;
  public sharedIndexBuffer: GPUBuffer;
  private sortedChunkGeometryInfoVertex: MemorySpaceInfo[];
  private sortedChunkGeometryInfoIndex: MemorySpaceInfo[];

  constructor(
    device: GPUDevice,
    sharedVertexBuffer: GPUBuffer,
    sharedIndexBuffer: GPUBuffer
  ) {
    this.device = device;
    this.sharedVertexBuffer = sharedVertexBuffer;
    this.sharedIndexBuffer = sharedIndexBuffer;
    this.chunkGeometryInfo = new Map();
    this.sortedChunkGeometryInfoVertex = [];
    this.sortedChunkGeometryInfoIndex = [];
  }

  getChunkGeometryInfo(position: vec3): ChunkGeometryInfo | undefined {
    const key = getChunkKey(position);
    return this.chunkGeometryInfo.get(key);
  }

  addChunk(
    position: vec3,
    vertexData: Float32Array,
    vertexSizeBytes: number,
    indexData: Uint32Array,
    indexSizeBytes: number,
    aabb: { min: vec3; max: vec3 }
  ) {
    const key = getChunkKey(position);

    console.log(
      `addChunk START for key: ${key}, vertexSize: ${vertexSizeBytes}, indexSize: ${indexSizeBytes}, indexDataLength: ${
        indexData?.byteLength ?? "null"
      }`
    );
    console.log(
      "Initial sortedChunkGeometryInfoVertex:",
      this.sortedChunkGeometryInfoVertex
    );
    console.log(
      "Initial sortedChunkGeometryInfoIndex:",
      this.sortedChunkGeometryInfoIndex
    );

    let foundVertex = false;
    let vertexOffsetBytes = 0;
    let i = 0;
    for (; i < this.sortedChunkGeometryInfoVertex.length - 1; i++) {
      const startOffset =
        this.sortedChunkGeometryInfoVertex[i].offset +
        this.sortedChunkGeometryInfoVertex[i].size;
      const spaceLeft =
        this.sortedChunkGeometryInfoVertex[i + 1].offset - startOffset;

      if (spaceLeft >= vertexSizeBytes) {
        foundVertex = true;
        vertexOffsetBytes = startOffset;
        this.sortedChunkGeometryInfoVertex.splice(i + 1, 0, {
          offset: vertexOffsetBytes,
          size: vertexSizeBytes,
          key,
        });
        break;
      }
    }
    if (!foundVertex) {
      if (this.sortedChunkGeometryInfoVertex.length === 0) {
        vertexOffsetBytes = 0;
      } else {
        const lastElementIndex = this.sortedChunkGeometryInfoVertex.length - 1;
        vertexOffsetBytes =
          this.sortedChunkGeometryInfoVertex[lastElementIndex].offset +
          this.sortedChunkGeometryInfoVertex[lastElementIndex].size;
      }

      this.sortedChunkGeometryInfoVertex.push({
        offset: vertexOffsetBytes,
        size: vertexSizeBytes,
        key,
      });
    }

    console.log(
      "SortedChunkGeometryInfoVertex:",
      JSON.stringify(this.sortedChunkGeometryInfoVertex)
    );

    console.log(
      `[Before Vertex Write] Type: ${vertexData?.constructor?.name}, Offset: ${vertexOffsetBytes}, Data Size: ${vertexData?.byteLength}, Buffer Size: ${this.sharedVertexBuffer?.size}, Buffer State: ${this.sharedVertexBuffer?.mapState}`
    );
    if (
      vertexOffsetBytes + vertexData.byteLength >
      this.sharedVertexBuffer.size
    ) {
      console.error("Attempting to write vertex data past buffer boundary!");
    }
    this.device.queue.writeBuffer(
      this.sharedVertexBuffer,
      vertexOffsetBytes,
      vertexData,
      0,
      vertexData.length
    );

    let foundIndex = false;
    let indexOffsetBytes = 0;
    i = 0;
    for (; i < this.sortedChunkGeometryInfoIndex.length - 1; i++) {
      const startOffset =
        this.sortedChunkGeometryInfoIndex[i].offset +
        this.sortedChunkGeometryInfoIndex[i].size;
      const spaceLeft =
        this.sortedChunkGeometryInfoIndex[i + 1].offset - startOffset;

      if (spaceLeft >= indexSizeBytes) {
        foundIndex = true;
        indexOffsetBytes = startOffset;
        this.sortedChunkGeometryInfoIndex.splice(i + 1, 0, {
          offset: indexOffsetBytes,
          size: indexSizeBytes,
          key,
        });
        break;
      }
    }
    if (!foundIndex) {
      if (this.sortedChunkGeometryInfoIndex.length === 0) {
        indexOffsetBytes = 0;
      } else {
        const lastElementIndex = this.sortedChunkGeometryInfoIndex.length - 1;
        indexOffsetBytes =
          this.sortedChunkGeometryInfoIndex[lastElementIndex].offset +
          this.sortedChunkGeometryInfoIndex[lastElementIndex].size;
      }

      this.sortedChunkGeometryInfoIndex.push({
        offset: indexOffsetBytes,
        size: indexSizeBytes,
        key,
      });
    }

    console.log(
      "SortedChunkGeometryInfoIndex:",
      this.sortedChunkGeometryInfoIndex
    );

    console.log(
      `[Before Index Write] Type: ${indexData?.constructor?.name}, Offset: ${indexOffsetBytes}, Data Size: ${indexData?.byteLength}, Buffer Size: ${this.sharedIndexBuffer?.size}, Buffer State: ${this.sharedIndexBuffer?.mapState}`
    );
    if (indexOffsetBytes + indexData.byteLength > this.sharedIndexBuffer.size) {
      console.error("Attempting to write index data past buffer boundary!");
    }
    if (indexData.byteLength > 0) {
      this.device.queue.writeBuffer(
        this.sharedIndexBuffer,
        indexOffsetBytes,
        indexData,
        0,
        indexData.length
      );
    } else {
      console.warn(
        `Index data for chunk ${key} is empty. Skipping index write.`
      );
    }

    const firstIndex =
      indexData.byteLength > 0 ? indexOffsetBytes / INDEX_SIZE_BYTES : 0;
    const baseVertex =
      vertexData.byteLength > 0 ? vertexOffsetBytes / VERTEX_STRIDE_BYTES : 0;

    if (firstIndex % 1 !== 0) {
      console.warn(
        `Calculated firstIndex (${firstIndex}) is not an integer. indexOffsetBytes=${indexOffsetBytes}, INDEX_SIZE_BYTES=${INDEX_SIZE_BYTES}`
      );
    }
    if (baseVertex % 1 !== 0) {
      console.warn(
        `Calculated baseVertex (${baseVertex}) is not an integer. vertexOffsetBytes=${vertexOffsetBytes}, VERTEX_STRIDE_BYTES=${VERTEX_STRIDE_BYTES}`
      );
    }

    this.chunkGeometryInfo.set(key, {
      indexCount: indexData.length,
      indexOffsetBytes,
      indexSizeBytes,
      indexData,
      vertexOffsetBytes,
      vertexSizeBytes,
      vertexData,
      aabb,
      position,
      firstIndex,
      baseVertex,
    });
  }

  deleteChunk(position: vec3) {
    const key = getChunkKey(position);
    this.freeChunkGeometryInfo(position);
    this.chunkGeometryInfo.delete(key);
  }

  updateChunkGeometryInfo(
    position: vec3,
    vertexData: Float32Array,
    vertexSizeBytes: number,
    indexData: Uint32Array,
    indexSizeBytes: number,
    aabb: { min: vec3; max: vec3 }
  ) {
    const key = getChunkKey(position);

    if (this.chunkGeometryInfo.has(key)) {
      this.freeChunkGeometryInfo(position);
    }

    this.addChunk(
      position,
      vertexData,
      vertexSizeBytes,
      indexData,
      indexSizeBytes,
      aabb
    );
  }

  freeChunkGeometryInfo(position: vec3) {
    const key = getChunkKey(position);
    const removed = this.chunkGeometryInfo.delete(key);

    if (removed) {
      let index: number;
      index = this.sortedChunkGeometryInfoVertex.findIndex(
        (v) => v.key === key
      );
      if (index !== -1) {
        this.sortedChunkGeometryInfoVertex.splice(index, 1);
      }
      index = this.sortedChunkGeometryInfoIndex.findIndex((v) => v.key === key);
      if (index !== -1) {
        this.sortedChunkGeometryInfoIndex.splice(index, 1);
      }
    }
  }
}
