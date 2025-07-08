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
            log("ChunkStorage", "Starting IndexedDB initialization...");
            
            // Check if IndexedDB is available (works in both main thread and workers)
            if (typeof indexedDB === 'undefined') {
                log.error("ChunkStorage", "IndexedDB is not available in this environment");
                reject(new Error("IndexedDB is not available in this environment"));
                return;
            }

            // Add timeout to prevent hanging
            const timeout = setTimeout(() => {
                log.error("ChunkStorage", "IndexedDB initialization timed out after 10 seconds");
                reject(new Error("IndexedDB initialization timed out after 10 seconds"));
            }, 10000);

            const attemptInit = () => {
                log("ChunkStorage", `Opening IndexedDB: ${this.dbName} v${this.version}`);
                const request = indexedDB.open(this.dbName, this.version);

                request.onerror = () => {
                    clearTimeout(timeout);
                    const error = request.error;
                    log.error("ChunkStorage", "Failed to open IndexedDB:", error);
                    
                    // Try to delete and recreate the database if it's corrupted
                    if (error && error.name === 'UnknownError') {
                        log("ChunkStorage", "Attempting to delete corrupted database...");
                        const deleteRequest = indexedDB.deleteDatabase(this.dbName);
                        
                        deleteRequest.onsuccess = () => {
                            log("ChunkStorage", "Database deleted, retrying initialization...");
                            setTimeout(() => {
                                attemptInit(); // Retry initialization
                            }, 100);
                        };
                        
                        deleteRequest.onerror = () => {
                            log.error("ChunkStorage", "Failed to delete corrupted database:", deleteRequest.error);
                            reject(error);
                        };
                    } else {
                        reject(error);
                    }
                };

                request.onsuccess = () => {
                    clearTimeout(timeout);
                    this.db = request.result;
                    this.isInitialized = true;
                    log("ChunkStorage", "IndexedDB initialized successfully");
                    resolve();
                };

                request.onupgradeneeded = (event) => {
                    log("ChunkStorage", "IndexedDB upgrade needed, creating object store...");
                    const db = (event.target as IDBOpenDBRequest).result;

                    // Create object store if it doesn't exist
                    if (!db.objectStoreNames.contains(this.storeName)) {
                        const store = db.createObjectStore(this.storeName, { keyPath: "key" });
                        store.createIndex("key", "key", { unique: true });
                        log("ChunkStorage", "Created chunk object store");
                    }
                };

                request.onblocked = () => {
                    log.error("ChunkStorage", "IndexedDB blocked - another connection is open");
                    reject(new Error("IndexedDB blocked - another connection is open"));
                };
            };

            // Start the initialization attempt
            attemptInit();
        });

        return this.initPromise;
    }

    async ensureInitialized(): Promise<void> {
        if (!this.isInitialized) {
            try {
                await this.initDatabase();
            } catch (error) {
                log.error("ChunkStorage", "Failed to initialize IndexedDB, storage will be disabled:", error);
                // Mark as initialized but with no database
                this.isInitialized = true;
                this.db = null;
            }
        }
    }

    async saveChunk(position: vec3, data: Uint8Array): Promise<void> {
        await this.ensureInitialized();

        if (!this.db) {
            // Storage is disabled, silently succeed
            return;
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
            // Storage is disabled, return null (chunk not found)
            return null;
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
            // Storage is disabled, silently succeed
            return;
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
            // Storage is disabled, return false (chunk not found)
            return false;
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
            // Storage is disabled, return empty stats
            const stats: ChunkStorageStats = {
                totalChunks: 0,
                totalSizeBytes: 0,
                averageChunkSizeBytes: 0
            };
            log("ChunkStorage", "Stats: Storage disabled (0 chunks)");
            return stats;
        }

        return new Promise((resolve, reject) => {
            const transaction = this.db!.transaction([this.storeName], "readonly");
            const store = transaction.objectStore(this.storeName);

            // Use count() instead of getAll() for much better performance
            const request = store.count();

            request.onsuccess = () => {
                const totalChunks = request.result;
                // Estimate size based on average chunk size (CHUNK_VOLUME bytes)
                const estimatedTotalSizeBytes = totalChunks * CHUNK_VOLUME;
                const averageChunkSizeBytes = CHUNK_VOLUME;

                const stats: ChunkStorageStats = {
                    totalChunks,
                    totalSizeBytes: estimatedTotalSizeBytes,
                    averageChunkSizeBytes
                };

                log("ChunkStorage", `Stats: ${totalChunks} chunks, ~${(estimatedTotalSizeBytes / 1024 / 1024).toFixed(2)} MB estimated`);
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