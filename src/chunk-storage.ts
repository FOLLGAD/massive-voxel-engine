import { vec3 } from "gl-matrix";
import { getChunkKey } from "./chunk";
import { CHUNK_VOLUME } from "./config";
import log from "./logger";

export interface ChunkStorageStats {
  totalChunks: number;
  totalSizeBytes: number;
  averageChunkSizeBytes: number;
}

export class ChunkStorage {
  private db: IDBDatabase | null = null;
  private readonly dbName = "VoxelEngineChunks";
  private readonly storeName = "chunks";
  private readonly version = 1;
  private isInitialized = false;
  private initPromise: Promise<void> | null = null;

  constructor() {
    this.initDatabase();
  }

  private async initDatabase(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.version);

      request.onerror = () => {
        log.error("ChunkStorage", "Failed to open IndexedDB:", request.error);
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.isInitialized = true;
        log("ChunkStorage", "IndexedDB initialized successfully");
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        
        // Create object store if it doesn't exist
        if (!db.objectStoreNames.contains(this.storeName)) {
          const store = db.createObjectStore(this.storeName, { keyPath: "key" });
          store.createIndex("key", "key", { unique: true });
          log("ChunkStorage", "Created chunk object store");
        }
      };
    });

    return this.initPromise;
  }

  private async ensureInitialized(): Promise<void> {
    if (!this.isInitialized) {
      await this.initDatabase();
    }
  }

  async saveChunk(position: vec3, data: Uint8Array): Promise<void> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      
      const chunkKey = getChunkKey(position);
      const chunkRecord = {
        key: chunkKey,
        position: Array.from(position),
        data: Array.from(data), // Convert to regular array for storage
        timestamp: Date.now(),
        size: data.byteLength
      };

      const request = store.put(chunkRecord);

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        log.error("ChunkStorage", `Failed to save chunk ${chunkKey}:`, request.error);
        reject(request.error);
      };
    });
  }

  async loadChunk(position: vec3): Promise<Uint8Array | null> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      
      const chunkKey = getChunkKey(position);
      const request = store.get(chunkKey);

      request.onsuccess = () => {
        if (request.result) {
          const chunkRecord = request.result;
          const data = new Uint8Array(chunkRecord.data);
          resolve(data);
        } else {
          resolve(null);
        }
      };

      request.onerror = () => {
        log.error("ChunkStorage", `Failed to load chunk ${chunkKey}:`, request.error);
        reject(request.error);
      };
    });
  }

  async deleteChunk(position: vec3): Promise<void> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      
      const chunkKey = getChunkKey(position);
      const request = store.delete(chunkKey);

      request.onsuccess = () => {
        resolve();
      };

      request.onerror = () => {
        log.error("ChunkStorage", `Failed to delete chunk ${chunkKey}:`, request.error);
        reject(request.error);
      };
    });
  }

  async hasChunk(position: vec3): Promise<boolean> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      
      const chunkKey = getChunkKey(position);
      const request = store.count(chunkKey);

      request.onsuccess = () => {
        resolve(request.result > 0);
      };

      request.onerror = () => {
        log.error("ChunkStorage", `Failed to check chunk ${chunkKey}:`, request.error);
        reject(request.error);
      };
    });
  }

  async getStats(): Promise<ChunkStorageStats> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      
      const request = store.getAll();

      request.onsuccess = () => {
        const chunks = request.result;
        const totalChunks = chunks.length;
        const totalSizeBytes = chunks.reduce((sum, chunk) => sum + chunk.size, 0);
        const averageChunkSizeBytes = totalChunks > 0 ? totalSizeBytes / totalChunks : 0;

        const stats: ChunkStorageStats = {
          totalChunks,
          totalSizeBytes,
          averageChunkSizeBytes
        };

        log("ChunkStorage", `Stats: ${totalChunks} chunks, ${(totalSizeBytes / 1024 / 1024).toFixed(2)} MB total`);
        resolve(stats);
      };

      request.onerror = () => {
        log.error("ChunkStorage", "Failed to get stats:", request.error);
        reject(request.error);
      };
    });
  }

  async clearAll(): Promise<void> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      
      const request = store.clear();

      request.onsuccess = () => {
        log("ChunkStorage", "Cleared all chunks from storage");
        resolve();
      };

      request.onerror = () => {
        log.error("ChunkStorage", "Failed to clear storage:", request.error);
        reject(request.error);
      };
    });
  }

  async getChunkKeys(): Promise<string[]> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      
      const request = store.getAllKeys();

      request.onsuccess = () => {
        const keys = request.result as string[];
        resolve(keys);
      };

      request.onerror = () => {
        log.error("ChunkStorage", "Failed to get chunk keys:", request.error);
        reject(request.error);
      };
    });
  }

  // Batch operations for better performance
  async saveChunks(chunks: Array<{ position: vec3; data: Uint8Array }>): Promise<void> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readwrite");
      const store = transaction.objectStore(this.storeName);
      
      let completed = 0;
      let hasError = false;

      const onComplete = () => {
        completed++;
        if (completed === chunks.length) {
          if (hasError) {
            reject(new Error("Some chunks failed to save"));
          } else {
            resolve();
          }
        }
      };

      chunks.forEach(({ position, data }) => {
        const chunkKey = getChunkKey(position);
        const chunkRecord = {
          key: chunkKey,
          position: Array.from(position),
          data: Array.from(data),
          timestamp: Date.now(),
          size: data.byteLength
        };

        const request = store.put(chunkRecord);

        request.onsuccess = onComplete;
        request.onerror = () => {
          log.error("ChunkStorage", `Failed to save chunk ${chunkKey} in batch:`, request.error);
          hasError = true;
          onComplete();
        };
      });
    });
  }

  async loadChunks(positions: vec3[]): Promise<Map<string, Uint8Array>> {
    await this.ensureInitialized();
    
    if (!this.db) {
      throw new Error("Database not initialized");
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.storeName], "readonly");
      const store = transaction.objectStore(this.storeName);
      
      const results = new Map<string, Uint8Array>();
      let completed = 0;

      const onComplete = () => {
        completed++;
        if (completed === positions.length) {
          resolve(results);
        }
      };

      positions.forEach(position => {
        const chunkKey = getChunkKey(position);
        const request = store.get(chunkKey);

        request.onsuccess = () => {
          if (request.result) {
            const chunkRecord = request.result;
            const data = new Uint8Array(chunkRecord.data);
            results.set(chunkKey, data);
          }
          onComplete();
        };

        request.onerror = () => {
          log.error("ChunkStorage", `Failed to load chunk ${chunkKey} in batch:`, request.error);
          onComplete();
        };
      });
    });
  }
} 