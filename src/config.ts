export const ENABLE_GREEDY_MESHING = true;
export const FLYING_SPEED = 100;
export const PHYSICS_FPS = 30;

export const LOAD_RADIUS_Y = 12;
export const LOAD_RADIUS_XZ = 16;

export const CHUNK_SIZE_X = 32;
export const CHUNK_SIZE_Y = 32;
export const CHUNK_SIZE_Z = 32;
export const CHUNK_VOLUME = CHUNK_SIZE_X * CHUNK_SIZE_Y * CHUNK_SIZE_Z;

export const UNLOAD_BUFFER_XZ = 4;
export const UNLOAD_BUFFER_Y = 1;

// Physics cache radius - how far to keep chunk data in RAM for collision detection
export const PHYSICS_CACHE_RADIUS_XZ = 3;
export const PHYSICS_CACHE_RADIUS_Y = 2;
