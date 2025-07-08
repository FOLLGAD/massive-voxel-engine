# Chunk Storage System

This document describes the file-based chunk storage system implemented for the massive voxel engine.

## Overview

The chunk storage system replaces the previous RAM-only storage with a hybrid approach that combines:
- **RAM Cache**: Fast access to frequently used chunks
- **IndexedDB Storage**: Persistent file storage for all chunks

## Architecture

### Components

1. **ChunkStorage** (`src/chunk-storage.ts`)
   - Handles IndexedDB operations
   - Provides async API for save/load/delete operations
   - Supports batch operations for better performance

2. **HybridChunkManager** (`src/hybrid-chunk-manager.ts`)
   - Manages both RAM cache and file storage
   - Implements LRU (Least Recently Used) cache eviction
   - Provides unified API for chunk operations

### Key Features

- **Persistent Storage**: Chunks are saved to IndexedDB and survive browser restarts
- **Smart Caching**: Frequently accessed chunks stay in RAM for fast access
- **Automatic Eviction**: When cache is full, least recently used chunks are evicted
- **Batch Operations**: Support for loading/saving multiple chunks at once
- **Error Handling**: Graceful fallback when storage operations fail

## Usage

### Basic Operations

```typescript
import { HybridChunkManager } from './src/hybrid-chunk-manager';
import { vec3 } from 'gl-matrix';

// Initialize with cache size (default: 1000 chunks)
const chunkManager = new HybridChunkManager(500);
await chunkManager.initialize();

// Save a chunk
const position = vec3.fromValues(1, 2, 3);
const chunkData = new Uint8Array(4096); // 16x16x16 chunk
await chunkManager.setChunk(position, chunkData);

// Load a chunk
const loadedData = await chunkManager.getChunk(position);

// Delete a chunk
await chunkManager.deleteChunk(position);

// Check if chunk exists
const exists = await chunkManager.hasChunk(position);
```

### Batch Operations

```typescript
// Save multiple chunks at once
const chunks = [
  { position: vec3.fromValues(1, 1, 1), data: new Uint8Array(4096) },
  { position: vec3.fromValues(1, 1, 2), data: new Uint8Array(4096) },
  // ...
];
await chunkManager.setChunks(chunks);

// Load multiple chunks at once
const positions = [vec3.fromValues(1, 1, 1), vec3.fromValues(1, 1, 2)];
const loadedChunks = await chunkManager.getChunks(positions);
```

### Statistics and Monitoring

```typescript
// Get performance statistics
const stats = await chunkManager.getStats();
console.log(`Cache hit rate: ${(stats.hitRate * 100).toFixed(1)}%`);
console.log(`Cache size: ${stats.cacheSize}/${stats.cacheHits + stats.cacheMisses}`);
console.log(`Storage chunks: ${stats.storageStats.totalChunks}`);
console.log(`Storage size: ${(stats.storageStats.totalSizeBytes / 1024 / 1024).toFixed(2)} MB`);
```

## Configuration

### Cache Size

The cache size determines how many chunks are kept in RAM:

```typescript
// Small cache (good for memory-constrained devices)
const chunkManager = new HybridChunkManager(100);

// Large cache (good for high-performance systems)
const chunkManager = new HybridChunkManager(2000);
```

### Storage Limits

IndexedDB storage limits vary by browser and available disk space:
- Chrome: ~80% of available disk space
- Firefox: ~50% of available disk space
- Safari: ~1GB by default

## Performance Considerations

### Cache Hit Rate

Monitor the cache hit rate to optimize cache size:
- High hit rate (>80%): Cache is working well
- Low hit rate (<50%): Consider increasing cache size

### Batch Operations

Use batch operations when loading/saving multiple chunks:
- Reduces IndexedDB transaction overhead
- Improves performance for bulk operations

### Memory Usage

Each cached chunk uses ~4KB of RAM (16x16x16 bytes):
- 100 chunks = ~400KB RAM
- 1000 chunks = ~4MB RAM
- 5000 chunks = ~20MB RAM

## Testing

Use the provided test page to verify the storage system:

```bash
# Open the test page
open storage-test.html
```

The test page includes:
- Basic save/load/delete operations
- Cache hit/miss testing
- Cache eviction testing
- Batch operations testing
- Performance statistics

## Integration with Main Application

The main application has been updated to use the hybrid chunk manager:

1. **Initialization**: Chunk manager is initialized during app startup
2. **Worker Integration**: Chunk data from workers is saved to storage
3. **Physics**: Physics system uses async chunk loading
4. **Block Placement**: Block modifications are persisted to storage
5. **Debug Info**: Press 'B' key to view storage statistics

## Migration from RAM-only Storage

The system automatically handles migration:
- Existing chunks in RAM are gradually moved to storage
- Cache provides backward compatibility
- No data loss during transition

## Troubleshooting

### Common Issues

1. **Storage Quota Exceeded**
   - Clear old chunks: `await chunkManager.clearAll()`
   - Reduce cache size
   - Check available disk space

2. **Slow Performance**
   - Increase cache size
   - Use batch operations
   - Monitor cache hit rate

3. **IndexedDB Errors**
   - Check browser console for errors
   - Verify IndexedDB is supported
   - Try clearing browser data

### Debug Commands

- Press 'B' in the main application to view storage stats
- Check browser console for detailed logs
- Use the test page for isolated testing

## Future Enhancements

Potential improvements:
- Compression for stored chunks
- Background chunk preloading
- Chunk versioning for updates
- Cloud storage integration
- Chunk streaming for large worlds 