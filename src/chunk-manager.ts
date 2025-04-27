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
    // try to find space for the vertex data
    let i = 0;
    // Loop should go up to length - 1 to compare element i with i + 1
    for (; i < this.sortedChunkGeometryInfoVertex.length - 1; i++) {
      // compare current offset + size with next offset to check how much space is left
      const startOffset =
        this.sortedChunkGeometryInfoVertex[i].offset +
        this.sortedChunkGeometryInfoVertex[i].size;
      const spaceLeft =
        this.sortedChunkGeometryInfoVertex[i + 1].offset - startOffset;

      if (spaceLeft >= vertexSizeBytes) {
        foundVertex = true;
        vertexOffsetBytes = startOffset;
        // insert the vertex data at the current offset
        // Use i + 1 as the insertion index because we found space *after* index i
        this.sortedChunkGeometryInfoVertex.splice(i + 1, 0, {
          offset: vertexOffsetBytes,
          size: vertexSizeBytes,
          key,
        });
        break;
      }
    }
    if (!foundVertex) {
      // If the array is empty, start at offset 0. Otherwise, append after the last chunk.
      if (this.sortedChunkGeometryInfoVertex.length === 0) {
        vertexOffsetBytes = 0;
      } else {
        // No need to recalculate i, the loop finishes with i pointing to the last checked index (or 0 if loop didn't run)
        // Append after the last element
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
      `[Staging Vertex Write] Offset: ${vertexOffsetBytes}, Data Size: ${vertexData.byteLength}, Target Buffer Size: ${this.sharedVertexBuffer?.size}`
    );

    // 1. Create Staging Buffer (ensure size is multiple of 4)
    // Make size slightly larger if needed to ensure alignment for copyBufferToBuffer sourceOffset (must be multiple of 4)
    const stagingVertexBufferSize = Math.ceil(vertexData.byteLength / 4) * 4;
    if (stagingVertexBufferSize === 0) {
      console.warn(`Vertex data for chunk ${key} is empty. Skipping vertex write.`);
    } else {
      const stagingVertexBuffer = this.device.createBuffer({
        label: `Staging Vertex Buffer for ${key}`,
        size: stagingVertexBufferSize,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, // Source for copy
        mappedAtCreation: true, // Map immediately
      });

      // 2. Write data to staging buffer's mapped range
      new Float32Array(stagingVertexBuffer.getMappedRange()).set(vertexData);
      stagingVertexBuffer.unmap();

      // 3. Create Command Encoder and Copy
      // Important: This needs to happen OUTSIDE the render pass encoding.
      // If addChunk is called during render prep, you might need to queue these commands
      // or create a separate encoder just for these copies.
      // Assuming we can create an encoder here for simplicity:
      const commandEncoder = this.device.createCommandEncoder({ label: `Vertex Copy Encoder for ${key}` });

      // Check if copy exceeds target buffer bounds BEFORE encoding
      if (vertexOffsetBytes + vertexData.byteLength > this.sharedVertexBuffer.size) {
        console.error(`[Copy Vertex] Attempting to copy vertex data past buffer boundary! Offset=${vertexOffsetBytes}, Size=${vertexData.byteLength}, BufferSize=${this.sharedVertexBuffer.size}`);
        // Decide how to handle - skip copy? throw error?
      } else {
        commandEncoder.copyBufferToBuffer(
          stagingVertexBuffer, // Source
          0, // Source Offset (must be multiple of 4)
          this.sharedVertexBuffer, // Destination
          vertexOffsetBytes, // Destination Offset (must be multiple of 4)
          vertexData.byteLength // Size to copy (must be multiple of 4 if not full buffer)
          // Using byteLength should be okay if it's already aligned,
          // otherwise use stagingVertexBufferSize if padding was added.
          // Let's stick to byteLength assuming input is correctly sized.
        );

        // 4. Submit copy command (needs to happen eventually)
        this.device.queue.submit([commandEncoder.finish()]);

        // 5. Destroy staging buffer (optional, but good practice after copy is submitted/done)
        // Note: Don't destroy immediately if submit hasn't happened or GPU hasn't finished.
        // A more robust system might pool/reuse staging buffers or destroy them later.
        // stagingVertexBuffer.destroy(); // Be careful with timing
      }
    }

    let foundIndex = false;
    let indexOffsetBytes = 0;

    i = 0;
    // Loop should go up to length - 1 to compare element i with i + 1
    for (; i < this.sortedChunkGeometryInfoIndex.length - 1; i++) {
      const startOffset =
        this.sortedChunkGeometryInfoIndex[i].offset +
        this.sortedChunkGeometryInfoIndex[i].size;
      const spaceLeft =
        this.sortedChunkGeometryInfoIndex[i + 1].offset - startOffset;

      if (spaceLeft >= indexSizeBytes) {
        foundIndex = true;
        indexOffsetBytes = startOffset;
        // Use i + 1 as the insertion index because we found space *after* index i
        this.sortedChunkGeometryInfoIndex.splice(i + 1, 0, {
          offset: indexOffsetBytes,
          size: indexSizeBytes,
          key,
        });
        break;
      }
    }
    if (!foundIndex) {
      // If the array is empty, start at offset 0. Otherwise, append after the last chunk.
      if (this.sortedChunkGeometryInfoIndex.length === 0) {
        indexOffsetBytes = 0;
      } else {
        // Append after the last element
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
      `[Staging Index Write] Offset: ${indexOffsetBytes}, Data Size: ${indexData.byteLength}, Target Buffer Size: ${this.sharedIndexBuffer?.size}`
    );

    // 1. Create Staging Buffer (ensure size is multiple of 4)
    const stagingIndexBufferSize = Math.ceil(indexData.byteLength / 4) * 4;
    if (stagingIndexBufferSize === 0) {
      console.warn(`Index data for chunk ${key} is empty. Skipping index write.`);
    } else {
      const stagingIndexBuffer = this.device.createBuffer({
        label: `Staging Index Buffer for ${key}`,
        size: stagingIndexBufferSize,
        usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });

      // 2. Write data to staging buffer
      new Uint32Array(stagingIndexBuffer.getMappedRange()).set(indexData); // Use Uint32Array for indices
      stagingIndexBuffer.unmap();

      // 3. Create Command Encoder and Copy
      // Reuse the vertex encoder or create a new one if needed
      // IMPORTANT: See previous note about command encoder timing
      const commandEncoder = this.device.createCommandEncoder({ label: `Index Copy Encoder for ${key}` });

      // Check bounds BEFORE encoding
      if (indexOffsetBytes + indexData.byteLength > this.sharedIndexBuffer.size) {
        console.error(`[Copy Index] Attempting to copy index data past buffer boundary! Offset=${indexOffsetBytes}, Size=${indexData.byteLength}, BufferSize=${this.sharedIndexBuffer.size}`);
      } else {
        commandEncoder.copyBufferToBuffer(
          stagingIndexBuffer,     // Source
          0,                      // Source Offset
          this.sharedIndexBuffer, // Destination
          indexOffsetBytes,        // Destination Offset
          indexData.byteLength     // Size
        );

        // 4. Submit copy command
        this.device.queue.submit([commandEncoder.finish()]);

        // 5. Destroy staging buffer (optional, consider timing)
        // stagingIndexBuffer.destroy();
      }
    }

    // Calculate firstIndex and baseVertex
    const firstIndex = indexData.byteLength > 0 ? indexOffsetBytes / INDEX_SIZE_BYTES : 0; // Avoid NaN if length is 0
    const baseVertex = vertexData.byteLength > 0 ? vertexOffsetBytes / VERTEX_STRIDE_BYTES : 0; // Avoid NaN if length is 0

    // Assert that calculations result in integers (important!)
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
    // Delete from the main map *first* so it's not found if accessed during freeing
    const removed = this.chunkGeometryInfo.delete(key);

    // Only try to splice if the key was actually present and removed
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
