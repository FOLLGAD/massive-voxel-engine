import { vec3 } from "gl-matrix";
import { getChunkKey } from "./chunk";
import { CHUNK_CONFIG } from "./config";
import log from "./logger";

interface WorldMetadata {
    worldName: string;
    worldSeed: number;
}

export interface ChunkStorageStats {
    totalChunks: number;
    totalSizeBytes: number;
    averageChunkSizeBytes: number;
}

export abstract class WorldStorageBase {
    abstract getWorlds(): Promise<string[]>;
    abstract loadWorld(worldId: string): Promise<ChunkStorageBase>;
    abstract createWorld(worldId: string, worldSeed: number, worldName: string): Promise<ChunkStorageBase>;
}

export class IdbWorldsStorage extends WorldStorageBase {
    private readonly worldDbPrefix = "VoxelEngineWorld:";
    private readonly version = 1;

    private openWorldDatabase(worldId: string, seedIfCreating?: number, worldName?: string): Promise<IDBDatabase> {
        const dbName = `${this.worldDbPrefix}${worldId}`;
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(dbName, this.version);

            request.onupgradeneeded = (event) => {
                const db = request.result;
                const existingStores = new Set(Array.from(db.objectStoreNames));
                if (!existingStores.has("chunks")) {
                    db.createObjectStore("chunks", { keyPath: "key" });
                }
                if (!existingStores.has("metadata")) {
                    const metaStore = db.createObjectStore("metadata", { keyPath: "key" });
                    // If we are explicitly creating a world, initialize metadata during upgrade
                    if (typeof seedIfCreating === "number") {
                        metaStore.put({ key: "world", worldSeed: seedIfCreating, worldName });
                    }
                }
            };

            request.onsuccess = () => {
                resolve(request.result);
            };
            request.onerror = () => {
                reject(request.error);
            };
        });
    }

    async getWorlds(): Promise<string[]> {
        // Prefer native databases() enumeration if available
        const anyIndexedDB = indexedDB as unknown as { databases?: () => Promise<Array<{ name?: string | null }>> };
        if (typeof anyIndexedDB.databases === "function") {
            try {
                const dbs = await anyIndexedDB.databases!();
                return dbs
                    .map((d) => d.name || "")
                    .filter((name) => name.startsWith(this.worldDbPrefix))
                    .map((name) => name.substring(this.worldDbPrefix.length));
            } catch (e) {
                log.warn("WorldsStorage", "indexedDB.databases() failed; returning empty world list", e);
                return [];
            }
        }
        // Fallback: cannot enumerate reliably without a registry; return empty list
        return [];
    }

    async loadWorld(worldId: string): Promise<ChunkStorageBase> {
        const db = await this.openWorldDatabase(worldId);
        const world = new IdbChunkStorage(db, "chunks", "metadata");
        await world.ensureInitialized();
        return world;
    }

    async createWorld(worldId: string, worldSeed: number, worldName: string): Promise<ChunkStorageBase> {
        const db = await this.openWorldDatabase(worldId, worldSeed, worldName);
        const world = new IdbChunkStorage(db, "chunks", "metadata");
        await world.ensureInitialized();
        return world;
    }
}

export abstract class ChunkStorageBase {
    abstract metadata: WorldMetadata | null;
    abstract saveChunk(position: vec3, data: Uint8Array, immediate?: boolean): Promise<void>;
    abstract loadChunk(position: vec3): Promise<Uint8Array | null>;
    abstract deleteChunk(position: vec3): Promise<void>;
    abstract hasChunk(position: vec3): Promise<boolean>;
    abstract getStats(): Promise<ChunkStorageStats>;
    abstract clearAll(): Promise<void>;
    abstract getChunkKeys(): Promise<string[]>;
    abstract saveChunks(chunks: Array<{ position: vec3; data: Uint8Array }>): Promise<void>;
    abstract loadChunks(positions: vec3[]): Promise<Map<string, Uint8Array>>;

    abstract ensureInitialized(): Promise<WorldMetadata>;
    abstract flushPendingSaves(): Promise<void>;
}

export class IdbChunkStorage extends ChunkStorageBase {
    private isInitialized = false;
    public metadata: WorldMetadata | null = null;

    // Batching mechanism for writes
    private pendingStorageSaves = new Map<string, { position: vec3; data: Uint8Array }>();
    private readonly STORAGE_BATCH_INTERVAL = 5000; // 5 seconds
    private readonly MAX_PENDING_SAVES = 100;
    private flushIntervalId: ReturnType<typeof setInterval> | null = null;

    constructor(private db: IDBDatabase, private readonly chunkStoreName: string, private readonly metadataStoreName: string) {
        super();

        // The batching system should only run on the main thread's instance.
        if (typeof window !== 'undefined') {
            this.flushIntervalId = setInterval(() => this.flushPendingSaves(), this.STORAGE_BATCH_INTERVAL);
            window.addEventListener('beforeunload', () => this.flushPendingSaves());
        }
    }

    private async initMetadata(): Promise<WorldMetadata> {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.metadataStoreName], "readonly");
            const store = transaction.objectStore(this.metadataStoreName);
            const request = store.get("world");
            request.onsuccess = () => {
                const metadata = request.result as WorldMetadata;
                this.metadata = metadata;
                resolve(metadata);
            };
            request.onerror = () => {
                log.error("ChunkStorage", "Failed to load metadata:", request.error);
                reject(new Error("Failed to load metadata"));
            };
        });
    }

    async ensureInitialized(): Promise<WorldMetadata> {
        if (!this.isInitialized) {
            const metadata = await this.initMetadata();
            this.isInitialized = true;
            return metadata;
        }
        return this.metadata!;
    }

    public async flushPendingSaves(): Promise<void> {
        if (this.pendingStorageSaves.size === 0) {
            return;
        }

        await this.ensureInitialized();
        if (!this.db) {
            log.warn("ChunkStorage", `Storage not available - discarding ${this.pendingStorageSaves.size} pending saves`);
            this.pendingStorageSaves.clear();
            return;
        }

        const chunksToSave = Array.from(this.pendingStorageSaves.values());
        this.pendingStorageSaves.clear();

        log("ChunkStorage", `💾 Flushing ${chunksToSave.length} chunks to storage.`);
        try {
            await this.saveChunks(chunksToSave);
            log("ChunkStorage", `✅ Successfully saved ${chunksToSave.length} chunks to IndexedDB storage`);
        } catch (error) {
            log.error("ChunkStorage", "❌ Failed to batch save chunks to storage:", error);
        }
    }

    async saveChunk(position: vec3, data: Uint8Array, immediate = false): Promise<void> {
        const chunkKey = getChunkKey(position);
        this.pendingStorageSaves.set(chunkKey, { position: vec3.clone(position), data: data.slice() });

        if (immediate || this.pendingStorageSaves.size >= this.MAX_PENDING_SAVES) {
            await this.flushPendingSaves();
        }
    }

    async loadChunk(position: vec3): Promise<Uint8Array | null> {
        await this.ensureInitialized();

        if (!this.db) {
            // Storage is disabled, return null (chunk not found)
            return null;
        }

        return new Promise((resolve, reject) => {
            const t = performance.now();
            const transaction = this.db.transaction([this.chunkStoreName], "readonly");
            const store = transaction.objectStore(this.chunkStoreName);

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
                console.log(`loadChunk took ${performance.now() - t}ms`);
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
            const transaction = this.db.transaction([this.chunkStoreName], "readwrite");
            const store = transaction.objectStore(this.chunkStoreName);

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
            const transaction = this.db.transaction([this.chunkStoreName], "readonly");
            const store = transaction.objectStore(this.chunkStoreName);

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
            const transaction = this.db.transaction([this.chunkStoreName], "readonly");
            const store = transaction.objectStore(this.chunkStoreName);

            // Use count() instead of getAll() for much better performance
            const request = store.count();

            request.onsuccess = () => {
                const totalChunks = request.result;
                // Estimate size based on average chunk size (CHUNK_VOLUME bytes)
                const estimatedTotalSizeBytes = totalChunks * CHUNK_CONFIG.volume;
                const averageChunkSizeBytes = CHUNK_CONFIG.volume;

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
            const transaction = this.db.transaction([this.chunkStoreName], "readwrite");
            const store = transaction.objectStore(this.chunkStoreName);

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
            const transaction = this.db.transaction([this.chunkStoreName], "readonly");
            const store = transaction.objectStore(this.chunkStoreName);

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
            return;
        }

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.chunkStoreName], "readwrite");
            const store = transaction.objectStore(this.chunkStoreName);

            let hasError = false;

            transaction.oncomplete = () => {
                if (hasError) {
                    reject(new Error("Some chunks failed to save in batch"));
                } else {
                    resolve();
                }
            };
            transaction.onerror = () => {
                reject(transaction.error);
            };

            chunks.forEach(({ position, data }) => {
                const chunkKey = getChunkKey(position);
                const chunkRecord = {
                    key: chunkKey,
                    position: Array.from(position),
                    data: data, // Storing Uint8Array directly
                    timestamp: Date.now(),
                    size: data.byteLength
                };

                const request = store.put(chunkRecord);
                request.onerror = () => {
                    log.error("ChunkStorage", `Failed to save chunk ${chunkKey} in batch:`, request.error);
                    hasError = true;
                };
            });
        });
    }

    async loadChunks(positions: vec3[]): Promise<Map<string, Uint8Array>> {
        await this.ensureInitialized();

        if (!this.db) {
            return new Map();
        }

        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([this.chunkStoreName], "readonly");
            const store = transaction.objectStore(this.chunkStoreName);
            const results = new Map<string, Uint8Array>();

            transaction.oncomplete = () => {
                resolve(results);
            };
            transaction.onerror = () => {
                reject(transaction.error);
            };

            positions.forEach(position => {
                const chunkKey = getChunkKey(position);
                const request = store.get(chunkKey);

                request.onsuccess = () => {
                    if (request.result) {
                        results.set(chunkKey, new Uint8Array(request.result.data));
                    }
                };
                request.onerror = () => {
                    log.error("ChunkStorage", `Failed to load chunk ${chunkKey} in batch:`, request.error);
                };
            });
        });
    }
} 