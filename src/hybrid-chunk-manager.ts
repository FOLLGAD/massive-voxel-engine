import { vec3 } from "gl-matrix";
import { getChunkKey } from "./chunk";
import { ChunkStorage } from "./chunk-storage";
import log from "./logger";

export interface ChunkCacheEntry {
  data: Uint8Array;
  lastAccessed: number;
  accessCount: number;
}

export interface HybridChunkManagerStats {
  cacheSize: number;
  cacheHits: number;
  cacheMisses: number;
  storageStats: any;
  hitRate: number;
}

export class HybridChunkManager {
  private storage: ChunkStorage;
  public cache: Map<string, ChunkCacheEntry>;
  private maxCacheSize: number;
  private cacheHits = 0;
  private cacheMisses = 0;
  private isInitialized = false;
  private pendingSaves = new Map<string, Uint8Array>();
  private saveTimeout: number | null = null;

  constructor(maxCacheSize = 1000) {
    this.storage = new ChunkStorage();
    this.cache = new Map();
    this.maxCacheSize = maxCacheSize;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;
    
    try {
      // Initialize storage
      await this.storage.getStats(); // This will trigger DB initialization
      this.isInitialized = true;
      log("HybridChunkManager", "Initialized successfully");
    } catch (error) {
      log.error("HybridChunkManager", "Failed to initialize:", error);
      throw error;
    }
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initialize();
    }
  }

  async getChunk(position: vec3): Promise<Uint8Array | null> {
    await this.ensureInitialized();
    
    const key = getChunkKey(position);
    const now = Date.now();

    // Check cache first
    const cached = this.cache.get(key);
    if (cached) {
      cached.lastAccessed = now;
      cached.accessCount++;
      this.cacheHits++;
      return cached.data;
    }

    this.cacheMisses++;

    // Try to load from storage
    try {
      const data = await this.storage.loadChunk(position);
      if (data) {
        // Add to cache
        this.addToCache(key, data);
        return data;
      } else {
        return null;
      }
    } catch (error) {
      log.error("HybridChunkManager", `Failed to load chunk ${key} from storage:`, error);
      return null;
    }
  }

  setChunk(position: vec3, data: Uint8Array): void {
    const key = getChunkKey(position);
    const now = Date.now();

    // Update cache
    this.addToCache(key, data);

    // Queue for batch save instead of immediate save
    this.pendingSaves.set(key, data);
    this.scheduleBatchSave();
  }

  private scheduleBatchSave(): void {
    if (this.saveTimeout) {
      clearTimeout(this.saveTimeout);
    }

    this.saveTimeout = setTimeout(() => {
      this.flushPendingSaves();
    }, 100) as any; // Batch saves every 100ms
  }

  private async flushPendingSaves(): Promise<void> {
    if (this.pendingSaves.size === 0) return;

    const chunks = Array.from(this.pendingSaves.entries()).map(([key, data]) => {
      const [x, y, z] = key.split(',').map(Number);
      return { position: vec3.fromValues(x, y, z), data };
    });

    this.pendingSaves.clear();

    // Save to storage asynchronously (don't wait for it)
    this.storage.saveChunks(chunks).catch(error => {
      log.error("HybridChunkManager", "Failed to save chunks in batch:", error);
    });
  }

  async deleteChunk(position: vec3): Promise<void> {
    await this.ensureInitialized();
    
    const key = getChunkKey(position);

    // Remove from cache
    this.cache.delete(key);

    // Remove from pending saves
    this.pendingSaves.delete(key);

    // Delete from storage
    try {
      await this.storage.deleteChunk(position);
    } catch (error) {
      log.error("HybridChunkManager", `Failed to delete chunk ${key} from storage:`, error);
    }
  }

  async hasChunk(position: vec3): Promise<boolean> {
    await this.ensureInitialized();
    
    const key = getChunkKey(position);

    // Check cache first
    if (this.cache.has(key)) {
      return true;
    }

    // Check storage
    return await this.storage.hasChunk(position);
  }

  private addToCache(key: string, data: Uint8Array): void {
    const now = Date.now();

    // If already in cache, update access info
    if (this.cache.has(key)) {
      const entry = this.cache.get(key)!;
      entry.lastAccessed = now;
      entry.accessCount++;
      return;
    }

    // If cache is full, evict least recently used
    if (this.cache.size >= this.maxCacheSize) {
      this.evictLRU();
    }

    // Add new entry
    this.cache.set(key, {
      data: data.slice(), // Make a copy to avoid shared references
      lastAccessed: now,
      accessCount: 1
    });
  }

  private evictLRU(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;
    let lowestAccessCount = Infinity;

    // Find the least recently used entry
    for (const [key, entry] of this.cache.entries()) {
      // Prioritize by access count first, then by last accessed time
      if (entry.accessCount < lowestAccessCount || 
          (entry.accessCount === lowestAccessCount && entry.lastAccessed < oldestTime)) {
        oldestKey = key;
        oldestTime = entry.lastAccessed;
        lowestAccessCount = entry.accessCount;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
    }
  }

  async clearCache(): Promise<void> {
    this.cache.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
    log("HybridChunkManager", "Cache cleared");
  }

  async clearAll(): Promise<void> {
    await this.ensureInitialized();
    
    this.cache.clear();
    this.cacheHits = 0;
    this.cacheMisses = 0;
    this.pendingSaves.clear();
    
    await this.storage.clearAll();
    log("HybridChunkManager", "All data cleared");
  }

  async getStats(): Promise<HybridChunkManagerStats> {
    await this.ensureInitialized();
    
    const storageStats = await this.storage.getStats();
    const totalRequests = this.cacheHits + this.cacheMisses;
    const hitRate = totalRequests > 0 ? this.cacheHits / totalRequests : 0;

    const stats: HybridChunkManagerStats = {
      cacheSize: this.cache.size,
      cacheHits: this.cacheHits,
      cacheMisses: this.cacheMisses,
      storageStats,
      hitRate
    };

    log("HybridChunkManager", `Stats: Cache ${this.cache.size}/${this.maxCacheSize}, Hit rate: ${(hitRate * 100).toFixed(1)}%`);
    return stats;
  }

  // Batch operations for better performance
  async getChunks(positions: vec3[]): Promise<Map<string, Uint8Array>> {
    await this.ensureInitialized();
    
    const results = new Map<string, Uint8Array>();
    const positionsToLoad: vec3[] = [];

    // Check cache first
    for (const position of positions) {
      const key = getChunkKey(position);
      const cached = this.cache.get(key);
      
      if (cached) {
        cached.lastAccessed = Date.now();
        cached.accessCount++;
        this.cacheHits++;
        results.set(key, cached.data);
      } else {
        this.cacheMisses++;
        positionsToLoad.push(position);
      }
    }

    // Load missing chunks from storage
    if (positionsToLoad.length > 0) {
      try {
        const loadedChunks = await this.storage.loadChunks(positionsToLoad);
        
        // Add loaded chunks to results and cache
        for (const [key, data] of loadedChunks.entries()) {
          results.set(key, data);
          this.addToCache(key, data);
        }
      } catch (error) {
        log.error("HybridChunkManager", "Failed to load chunks in batch:", error);
      }
    }

    return results;
  }

  setChunks(chunks: Array<{ position: vec3; data: Uint8Array }>): void {
    // Update cache
    for (const { position, data } of chunks) {
      const key = getChunkKey(position);
      this.addToCache(key, data);
      this.pendingSaves.set(key, data);
    }

    this.scheduleBatchSave();
  }

  // Preload chunks into cache
  async preloadChunks(positions: vec3[]): Promise<void> {
    await this.ensureInitialized();
    
    const chunksToLoad: vec3[] = [];
    
    // Filter out chunks already in cache
    for (const position of positions) {
      const key = getChunkKey(position);
      if (!this.cache.has(key)) {
        chunksToLoad.push(position);
      }
    }

    if (chunksToLoad.length === 0) {
      return;
    }

    try {
      const loadedChunks = await this.storage.loadChunks(chunksToLoad);
      
      // Add to cache
      for (const [key, data] of loadedChunks.entries()) {
        this.addToCache(key, data);
      }
      
      log("HybridChunkManager", `Preloaded ${loadedChunks.size} chunks`);
    } catch (error) {
      log.error("HybridChunkManager", "Failed to preload chunks:", error);
    }
  }

  // Get cache keys for debugging
  getCacheKeys(): string[] {
    return Array.from(this.cache.keys());
  }

  // Set cache size limit
  setMaxCacheSize(size: number): void {
    this.maxCacheSize = size;
    
    // Evict excess entries if needed
    while (this.cache.size > this.maxCacheSize) {
      this.evictLRU();
    }
  }

  // Force flush pending saves (for shutdown)
  async flush(): Promise<void> {
    await this.flushPendingSaves();
  }
} 