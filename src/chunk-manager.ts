import { getChunkKey } from "./chunk";
import type { AABB } from "./aabb";
import type { vec3 } from "gl-matrix";

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
  aabb: AABB;

  /** The position of the chunk. */
  position: vec3;

  /** The offset (in indices) into the shared index buffer for the first index. */
  firstIndex: number;

  /** The value added to each index before reading from the vertex buffer. */
  baseVertex: number;

  /** The visibility bits for the chunk. */
  visibilityBits: number;
}

export interface MemorySpaceInfo {
  offset: number;
  size: number;
  key: string;
}

// Interface for free blocks (no key needed)
export interface FreeSpaceInfo {
  offset: number;
  size: number;
}

// Helper function to insert a block into a sorted free list and merge adjacent blocks
function insertAndMergeFreeBlock(
  freeList: FreeSpaceInfo[],
  newBlock: FreeSpaceInfo
) {
  let inserted = false;
  for (let i = 0; i < freeList.length; i++) {
    if (newBlock.offset < freeList[i].offset) {
      // Insert before element i
      freeList.splice(i, 0, newBlock);
      inserted = true;

      // Attempt merge with previous (if exists and adjacent)
      if (
        i > 0 &&
        freeList[i - 1].offset + freeList[i - 1].size === newBlock.offset
      ) {
        freeList[i - 1].size += newBlock.size;
        freeList.splice(i, 1); // Remove newBlock (now merged)
        i--; // Adjust index after removal
      }

      // Attempt merge with next (element at new index i, which was originally at i or i+1)
      const currentBlock = freeList[i];
      if (
        i + 1 < freeList.length &&
        currentBlock.offset + currentBlock.size === freeList[i + 1].offset
      ) {
        currentBlock.size += freeList[i + 1].size;
        freeList.splice(i + 1, 1); // Remove next block
      }
      break; // Exit loop once inserted and merged
    }
  }

  // If not inserted yet, it belongs at the end
  if (!inserted) {
    // Attempt merge with last element (if exists and adjacent)
    if (
      freeList.length > 0 &&
      freeList[freeList.length - 1].offset +
      freeList[freeList.length - 1].size ===
      newBlock.offset
    ) {
      freeList[freeList.length - 1].size += newBlock.size;
    } else {
      freeList.push(newBlock);
    }
  }
}

export class ChunkManager {
  chunkGeometryInfo: Map<string, ChunkGeometryInfo>;

  public device: GPUDevice;
  public sharedVertexBuffer: GPUBuffer;
  public sharedIndexBuffer: GPUBuffer;

  // Track used space using Maps for fast key lookup
  private usedSpaceVertex: Map<string, MemorySpaceInfo>;
  private usedSpaceIndex: Map<string, MemorySpaceInfo>;

  // Track the end of the last appended block for fast append offset calculation
  private maxVertexOffsetEnd: number;
  private maxIndexOffsetEnd: number;

  // Track free space (sorted by offset)
  private freeSpaceVertex: FreeSpaceInfo[];
  private freeSpaceIndex: FreeSpaceInfo[];

  // Optional renderer reference for cache invalidation
  private renderer?: { invalidateCullCache(): void };

  constructor(
    device: GPUDevice,
    sharedVertexBuffer: GPUBuffer,
    sharedIndexBuffer: GPUBuffer,
    renderer?: { invalidateCullCache(): void }
  ) {
    this.device = device;
    this.sharedVertexBuffer = sharedVertexBuffer;
    this.sharedIndexBuffer = sharedIndexBuffer;
    this.chunkGeometryInfo = new Map();
    this.renderer = renderer;

    // Initialize used space maps
    this.usedSpaceVertex = new Map();
    this.usedSpaceIndex = new Map();

    // Initialize append point trackers
    this.maxVertexOffsetEnd = 0;
    this.maxIndexOffsetEnd = 0;

    // Initialize free space lists
    this.freeSpaceVertex = [{ offset: 0, size: sharedVertexBuffer.size }];
    this.freeSpaceIndex = [{ offset: 0, size: sharedIndexBuffer.size }];
  }

  getChunkGeometryInfo(position: vec3): ChunkGeometryInfo | undefined {
    const key = getChunkKey(position);
    return this.chunkGeometryInfo.get(key);
  }

  // Set the renderer reference for cache invalidation
  setRenderer(renderer: { invalidateCullCache(): void }): void {
    this.renderer = renderer;
  }

  addChunk(
    position: vec3,
    vertexData: Float32Array,
    vertexSizeBytes: number,
    indexData: Uint32Array,
    indexSizeBytes: number,
    aabb: AABB,
    visibilityBits: number
  ) {
    const key = getChunkKey(position);

    const chunkGeometryInfo: ChunkGeometryInfo = {
      indexCount: 0, // Placeholder
      indexOffsetBytes: -1, // Placeholder
      indexSizeBytes: indexSizeBytes,
      indexData: indexData, // Keep data reference
      vertexOffsetBytes: -1, // Placeholder
      vertexSizeBytes: vertexSizeBytes,
      vertexData: vertexData, // Keep data reference
      aabb: aabb,
      position: position,
      firstIndex: -1, // Placeholder
      baseVertex: -1, // Placeholder
      visibilityBits,
    };
    this.chunkGeometryInfo.set(key, chunkGeometryInfo);

    let vertexOffsetBytes = -1;
    let indexOffsetBytes = -1;
    let usedExistingFreeVertexBlock = false;
    let usedExistingFreeIndexBlock = false;

    // --- Allocate Vertex Space ---
    for (let i = 0; i < this.freeSpaceVertex.length; i++) {
      const freeBlock = this.freeSpaceVertex[i];
      if (freeBlock.size >= vertexSizeBytes) {
        vertexOffsetBytes = freeBlock.offset;
        usedExistingFreeVertexBlock = true;
        if (freeBlock.size === vertexSizeBytes) {
          this.freeSpaceVertex.splice(i, 1);
        } else {
          freeBlock.offset += vertexSizeBytes;
          freeBlock.size -= vertexSizeBytes;
        }
        break;
      }
    }

    // If no free space found, append
    if (vertexOffsetBytes === -1) {
      vertexOffsetBytes = this.maxVertexOffsetEnd; // Use the tracked end point
      this.maxVertexOffsetEnd += vertexSizeBytes; // Increment for next append
    }

    // Store used block info in Map
    const newUsedVertexBlock: MemorySpaceInfo = {
      offset: vertexOffsetBytes,
      size: vertexSizeBytes,
      key,
    };
    this.usedSpaceVertex.set(key, newUsedVertexBlock);

    // --- Allocate Index Space ---
    for (let i = 0; i < this.freeSpaceIndex.length; i++) {
      const freeBlock = this.freeSpaceIndex[i];
      if (freeBlock.size >= indexSizeBytes) {
        indexOffsetBytes = freeBlock.offset;
        usedExistingFreeIndexBlock = true;
        if (freeBlock.size === indexSizeBytes) {
          this.freeSpaceIndex.splice(i, 1);
        } else {
          freeBlock.offset += indexSizeBytes;
          freeBlock.size -= indexSizeBytes;
        }
        break;
      }
    }

    // If no free space found, append
    if (indexOffsetBytes === -1) {
      indexOffsetBytes = this.maxIndexOffsetEnd; // Use tracked end point
      this.maxIndexOffsetEnd += indexSizeBytes; // Increment for next append
    }

    // Store used block info in Map
    const newUsedIndexBlock: MemorySpaceInfo = {
      offset: indexOffsetBytes,
      size: indexSizeBytes,
      key,
    };
    this.usedSpaceIndex.set(key, newUsedIndexBlock);

    // --- Error Handling Helper ---
    // Function to clean up state if a write fails
    const handleWriteError = (bufferType: "vertex" | "index") => {
      console.error(
        `Error during ${bufferType} write for chunk ${key}. Cleaning up.`
      );
      // Remove from used space maps
      this.usedSpaceVertex.delete(key);
      this.usedSpaceIndex.delete(key);
      // Attempt to roll back allocation
      if (usedExistingFreeVertexBlock) {
        insertAndMergeFreeBlock(this.freeSpaceVertex, {
          offset: vertexOffsetBytes,
          size: vertexSizeBytes,
        });
      } else {
        this.maxVertexOffsetEnd -= vertexSizeBytes;
      }
      if (usedExistingFreeIndexBlock) {
        insertAndMergeFreeBlock(this.freeSpaceIndex, {
          offset: indexOffsetBytes,
          size: indexSizeBytes,
        });
      } else {
        this.maxIndexOffsetEnd -= indexSizeBytes;
      }
      // Remove the preliminary entry from the main map
      this.chunkGeometryInfo.delete(key);
    };

    // --- Write Data to Buffers ---
    // Vertex Write
    if (
      vertexOffsetBytes + vertexData.byteLength >
      this.sharedVertexBuffer.size
    ) {
      console.error(`Vertex write buffer overflow for key ${key}!`);
      handleWriteError("vertex");
      return;
    }
    try {
      if (vertexData.byteLength > 0) {
        this.device.queue.writeBuffer(
          this.sharedVertexBuffer,
          vertexOffsetBytes,
          vertexData,
          0,
          vertexData.length
        );
      }
    } catch (error) {
      console.error(
        `Caught exception during vertex writeBuffer for ${key}:`,
        error
      );
      handleWriteError("vertex");
      return;
    }

    // Index Write
    if (indexOffsetBytes + indexData.byteLength > this.sharedIndexBuffer.size) {
      console.error(`Index write buffer overflow for key ${key}!`);
      handleWriteError("index"); // Also roll back vertex allocation here
      return;
    }
    try {
      if (indexData.byteLength > 0) {
        this.device.queue.writeBuffer(
          this.sharedIndexBuffer,
          indexOffsetBytes,
          indexData,
          0,
          indexData.length
        );
      }
    } catch (error) {
      console.error(
        `Caught exception during index writeBuffer for ${key}:`,
        error
      );
      handleWriteError("index");
      return;
    }
    // --- End Write Data ---

    // Calculate firstIndex and baseVertex
    const firstIndex =
      indexData.byteLength > 0 ? indexOffsetBytes / INDEX_SIZE_BYTES : 0;
    const baseVertex =
      vertexData.byteLength > 0 ? vertexOffsetBytes / VERTEX_STRIDE_BYTES : 0;
    if (firstIndex % 1 !== 0)
      console.warn(`Non-integer firstIndex: ${firstIndex}`);
    if (baseVertex % 1 !== 0)
      console.warn(`Non-integer baseVertex: ${baseVertex}`);

    // --- Update the placeholder geometry info in the map ---
    // !! DO NOT set status to ready here !!
    if (chunkGeometryInfo) {
      chunkGeometryInfo.indexCount = indexData.length;
      chunkGeometryInfo.indexOffsetBytes = indexOffsetBytes;
      chunkGeometryInfo.vertexOffsetBytes = vertexOffsetBytes;
      chunkGeometryInfo.firstIndex = firstIndex;
      chunkGeometryInfo.baseVertex = baseVertex;
      // finalChunkInfo.status remains 'updating'
    } else {
      console.error(`Chunk info for key ${key} missing after allocation!`);
    }

    // Invalidate renderer culling cache
    this.renderer?.invalidateCullCache();
  }

  deleteChunk(position: vec3) {
    this.freeChunkGeometryInfo(position);
    this.chunkGeometryInfo.delete(getChunkKey(position));
    
    // Invalidate renderer culling cache
    this.renderer?.invalidateCullCache();
  }

  updateChunkGeometryInfo(
    position: vec3,
    vertexData: Float32Array,
    vertexSizeBytes: number,
    indexData: Uint32Array,
    indexSizeBytes: number,
    aabb: AABB,
    visibilityBits: number
  ) {
    const key = getChunkKey(position);
    const existingChunkInfo = this.chunkGeometryInfo.get(key);

    if (!existingChunkInfo) {
      this.addChunk(
        position,
        vertexData,
        vertexSizeBytes,
        indexData,
        indexSizeBytes,
        aabb,
        visibilityBits
      );
      return;
    }

    existingChunkInfo.position = position;
    existingChunkInfo.aabb = aabb;
    existingChunkInfo.vertexData = vertexData;
    existingChunkInfo.indexData = indexData;
    existingChunkInfo.vertexSizeBytes = vertexSizeBytes;
    existingChunkInfo.indexSizeBytes = indexSizeBytes;

    // Store old offsets/sizes before freeing
    const oldVertexOffset = existingChunkInfo.vertexOffsetBytes;
    const oldVertexSize = this.usedSpaceVertex.get(key)?.size;
    const oldIndexOffset = existingChunkInfo.indexOffsetBytes;
    const oldIndexSize = this.usedSpaceIndex.get(key)?.size;

    // Free old vertex space
    if (oldVertexSize !== undefined) {
      if (this.usedSpaceVertex.delete(key)) {
        insertAndMergeFreeBlock(this.freeSpaceVertex, {
          offset: oldVertexOffset,
          size: oldVertexSize,
        });
      } else {
        console.warn(`[Update] Vertex delete failed for ${key}`);
      }
    } else if (oldVertexOffset !== -1) {
      console.warn(
        `[Update] oldVertexSize undefined but offset existed for ${key}`
      );
    }

    // Free old index space
    if (oldIndexSize !== undefined) {
      if (this.usedSpaceIndex.delete(key)) {
        insertAndMergeFreeBlock(this.freeSpaceIndex, {
          offset: oldIndexOffset,
          size: oldIndexSize,
        });
      } else {
        console.warn(`[Update] Index delete failed for ${key}`);
      }
    } else if (oldIndexOffset !== -1) {
      console.warn(
        `[Update] oldIndexSize undefined but offset existed for ${key}`
      );
    }

    // Allocate NEW space
    let vertexOffsetBytes = -1;
    let indexOffsetBytes = -1;
    let usedExistingFreeVertexBlock = false;
    let usedExistingFreeIndexBlock = false;

    // Vertex allocation (find free or append)
    for (let i = 0; i < this.freeSpaceVertex.length; i++) {
      const freeBlock = this.freeSpaceVertex[i];
      if (freeBlock.size >= vertexSizeBytes) {
        vertexOffsetBytes = freeBlock.offset;
        usedExistingFreeVertexBlock = true;
        if (freeBlock.size === vertexSizeBytes)
          this.freeSpaceVertex.splice(i, 1);
        else {
          freeBlock.offset += vertexSizeBytes;
          freeBlock.size -= vertexSizeBytes;
        }
        break;
      }
    }
    if (vertexOffsetBytes === -1) {
      vertexOffsetBytes = this.maxVertexOffsetEnd;
      this.maxVertexOffsetEnd += vertexSizeBytes;
    }
    this.usedSpaceVertex.set(key, {
      offset: vertexOffsetBytes,
      size: vertexSizeBytes,
      key,
    });

    // Index allocation (find free or append)
    for (let i = 0; i < this.freeSpaceIndex.length; i++) {
      const freeBlock = this.freeSpaceIndex[i];
      if (freeBlock.size >= indexSizeBytes) {
        indexOffsetBytes = freeBlock.offset;
        usedExistingFreeIndexBlock = true;
        if (freeBlock.size === indexSizeBytes) this.freeSpaceIndex.splice(i, 1);
        else {
          freeBlock.offset += indexSizeBytes;
          freeBlock.size -= indexSizeBytes;
        }
        break;
      }
    }
    if (indexOffsetBytes === -1) {
      indexOffsetBytes = this.maxIndexOffsetEnd;
      this.maxIndexOffsetEnd += indexSizeBytes;
    }
    this.usedSpaceIndex.set(key, {
      offset: indexOffsetBytes,
      size: indexSizeBytes,
      key,
    });

    if (
      vertexOffsetBytes + vertexData.byteLength >
      this.sharedVertexBuffer.size
    ) {
      console.error(`[Update] Vertex write buffer overflow for key ${key}!`);
      return;
    }
    try {
      if (vertexData.byteLength > 0)
        this.device.queue.writeBuffer(
          this.sharedVertexBuffer,
          vertexOffsetBytes,
          vertexData,
          0,
          vertexData.length
        );
    } catch (error) {
      console.error("[Update] Vertex write error:", error);
      return;
    }

    // Index Write
    if (indexOffsetBytes + indexData.byteLength > this.sharedIndexBuffer.size) {
      console.error(`[Update] Index write buffer overflow for key ${key}!`);
      return;
    }
    try {
      if (indexData.byteLength > 0)
        this.device.queue.writeBuffer(
          this.sharedIndexBuffer,
          indexOffsetBytes,
          indexData,
          0,
          indexData.length
        );
    } catch (error) {
      console.error("[Update] Index write error:", error);
      return;
    }

    // Update existing chunkInfo with final details
    // !! DO NOT set status to ready here !!
    const finalFirstIndex =
      indexData.byteLength > 0 ? indexOffsetBytes / INDEX_SIZE_BYTES : 0;
    const finalBaseVertex =
      vertexData.byteLength > 0 ? vertexOffsetBytes / VERTEX_STRIDE_BYTES : 0;
    if (finalFirstIndex % 1 !== 0)
      console.warn(`[Update] Non-integer firstIndex: ${finalFirstIndex}`);
    if (finalBaseVertex % 1 !== 0)
      console.warn(`[Update] Non-integer baseVertex: ${finalBaseVertex}`);

    existingChunkInfo.indexCount = indexData.length;
    existingChunkInfo.indexOffsetBytes = indexOffsetBytes;
    existingChunkInfo.vertexOffsetBytes = vertexOffsetBytes;
    existingChunkInfo.firstIndex = finalFirstIndex;
    existingChunkInfo.baseVertex = finalBaseVertex;
    existingChunkInfo.visibilityBits = visibilityBits;

    // Invalidate renderer culling cache
    this.renderer?.invalidateCullCache();
  }

  freeChunkGeometryInfo(position: vec3) {
    const key = getChunkKey(position);
    const chunkInfo = this.chunkGeometryInfo.get(key);

    if (!chunkInfo) {
      // Don't warn if just updating non-existent, could be intentional
      // console.warn(`Attempted to free non-existent chunk: ${key}`);
      return; // Chunk doesn't exist
    }

    // 1. Remove from chunkGeometryInfo map
    this.chunkGeometryInfo.delete(key);

    // 2. Get used block info from Maps, remove, add/merge to freeSpace
    const usedVertexBlock = this.usedSpaceVertex.get(key);
    if (usedVertexBlock) {
      this.usedSpaceVertex.delete(key); // Remove from map
      insertAndMergeFreeBlock(this.freeSpaceVertex, {
        offset: usedVertexBlock.offset,
        size: usedVertexBlock.size,
      });
    } else {
      console.warn(
        `Could not find vertex block for key ${key} in usedSpaceVertex map during free.`
      );
    }

    const usedIndexBlock = this.usedSpaceIndex.get(key);
    if (usedIndexBlock) {
      this.usedSpaceIndex.delete(key); // Remove from map
      insertAndMergeFreeBlock(this.freeSpaceIndex, {
        offset: usedIndexBlock.offset,
        size: usedIndexBlock.size,
      });
    } else {
      console.warn(
        `Could not find index block for key ${key} in usedSpaceIndex map during free.`
      );
    }
  }
}
