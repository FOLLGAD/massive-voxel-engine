export const ENABLE_GREEDY_MESHING = true;
export const FLYING_SPEED = 100;
export const PHYSICS_FPS = 30;

export const CHUNK_CONFIG = {
    loadRadius: {
        xz: 8,
        y: 8
    },
    size: {
        x: 32,
        y: 32,
        z: 32
    },
    get volume(): number {
        return this.size.x * this.size.y * this.size.z;
    }
}

export const UNLOAD_BUFFER_XZ = 4;
export const UNLOAD_BUFFER_Y = 1;

// Physics cache radius - how far to keep chunk data in RAM for collision detection
export const PHYSICS_CACHE_RADIUS_XZ = 3;
export const PHYSICS_CACHE_RADIUS_Y = 2;
