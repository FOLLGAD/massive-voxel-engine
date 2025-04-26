import { vec3 } from "gl-matrix";
import { CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z } from "./common/constants";
import log from "./logger";
import { Chunk, getChunkKey } from "./chunk";
import { ENABLE_FLYING_MODE } from "./config";

// --- Physics Constants ---
export const GRAVITY = -16.0; // Units per second squared
export const JUMP_VELOCITY = 7.0; // Initial upward velocity on jump
export const MAX_STEP_HEIGHT = 0.75; // How high the player can step up automatically
export const PLAYER_HEIGHT = 1.87;
export const PLAYER_WIDTH = 0.8; // Includes depth
export const MOVE_SPEED = 5.0; // Units per second
export const PLAYER_HALF_WIDTH = PLAYER_WIDTH / 2;
const COLLISION_EPSILON = 1e-6;
const PLAYER_EYE_LEVEL = 1.6;

export interface AABB {
  min: vec3;
  max: vec3;
}

function createAABB(
  minX: number,
  minY: number,
  minZ: number,
  maxX: number,
  maxY: number,
  maxZ: number
): AABB {
  return {
    min: vec3.fromValues(minX, minY, minZ),
    max: vec3.fromValues(maxX, maxY, maxZ),
  };
}

function getPlayerAABB(position: vec3): AABB {
  return createAABB(
    position[0] - PLAYER_HALF_WIDTH,
    position[1],
    position[2] - PLAYER_HALF_WIDTH,
    position[0] + PLAYER_HALF_WIDTH,
    position[1] + PLAYER_HEIGHT,
    position[2] + PLAYER_HALF_WIDTH
  );
}

function expandAABB(aabb: AABB, x: number, y: number, z: number): AABB {
  const newMin = vec3.clone(aabb.min);
  const newMax = vec3.clone(aabb.max);
  if (x < 0) newMin[0] += x;
  else newMax[0] += x;
  if (y < 0) newMin[1] += y;
  else newMax[1] += y;
  if (z < 0) newMin[2] += z;
  else newMax[2] += z;
  return { min: newMin, max: newMax };
}

function doesAABBOverlap(a: AABB, b: AABB): boolean {
  return (
    a.min[0] < b.max[0] &&
    a.max[0] > b.min[0] &&
    a.min[1] < b.max[1] &&
    a.max[1] > b.min[1] &&
    a.min[2] < b.max[2] &&
    a.max[2] > b.min[2]
  );
}

function getVoxelAABB(voxelX: number, voxelY: number, voxelZ: number): AABB {
  return createAABB(voxelX, voxelY, voxelZ, voxelX + 1, voxelY + 1, voxelZ + 1);
}

// --- Player State ---
export class PlayerState {
  position: vec3;
  velocity: vec3;
  isGrounded: boolean;

  constructor() {
    this.position = vec3.fromValues(0, 48, 0);
    this.velocity = vec3.create();
    this.isGrounded = false;
  }

  static fromValues(
    position: vec3,
    velocity: vec3,
    isGrounded: boolean
  ): PlayerState {
    const playerState = new PlayerState();
    playerState.position = vec3.clone(position);
    playerState.velocity = vec3.clone(velocity);
    playerState.isGrounded = isGrounded;
    return playerState;
  }

  getCameraPosition(): vec3 {
    return vec3.fromValues(
      this.position[0],
      this.position[1] + PLAYER_EYE_LEVEL,
      this.position[2]
    );
  }
}

// --- Voxel Interaction ---

// Check if a voxel type is solid
function isSolidVoxel(voxelType: number): boolean {
  return voxelType !== 0; // Assuming 0 is Air
}

// Get voxel type at world coordinates (Needs access to loadedChunkData)
function getVoxelAt(
  worldX: number,
  worldY: number,
  worldZ: number,
  loadedChunkData: Map<string, Uint8Array>
): number {
  const chunkX = Math.floor(worldX / CHUNK_SIZE_X);
  const chunkY = Math.floor(worldY / CHUNK_SIZE_Y);
  const chunkZ = Math.floor(worldZ / CHUNK_SIZE_Z);

  const key = getChunkKey({ x: chunkX, y: chunkY, z: chunkZ });
  const chunkData = loadedChunkData.get(key);

  if (!chunkData) {
    log.warn(
      "Physics",
      `Chunk data not loaded for ${key} at getVoxelAt(${worldX.toFixed(
        1
      )}, ${worldY.toFixed(1)}, ${worldZ.toFixed(1)})`
    );
    return 0; // Treat unloaded chunks as Air for safety
  }

  const chunk = Chunk.withData(
    {
      x: chunkX,
      y: chunkY,
      z: chunkZ,
    },
    chunkData
  );

  const voxelX = Math.floor(worldX) - chunk.position.x * CHUNK_SIZE_X;
  const voxelY = Math.floor(worldY) - chunk.position.y * CHUNK_SIZE_Y;
  const voxelZ = Math.floor(worldZ) - chunk.position.z * CHUNK_SIZE_Z;

  const voxel = chunk.getVoxel(voxelX, voxelY, voxelZ);

  return voxel;
}

/**
 * Finds all solid voxel AABBs that potentially collide with a given AABB.
 * This serves as the broad phase collision detection.
 */
function getPotentialVoxelCollisions(
  aabb: AABB,
  loadedChunkData: Map<string, Uint8Array>
): AABB[] {
  const collisions: AABB[] = [];
  const minX = Math.floor(aabb.min[0]);
  const minY = Math.floor(aabb.min[1]);
  const minZ = Math.floor(aabb.min[2]);
  const maxX = Math.ceil(aabb.max[0]);
  const maxY = Math.ceil(aabb.max[1]);
  const maxZ = Math.ceil(aabb.max[2]);

  for (let y = minY; y < maxY; y++) {
    for (let z = minZ; z < maxZ; z++) {
      for (let x = minX; x < maxX; x++) {
        const voxelType = getVoxelAt(
          x + 0.5,
          y + 0.5,
          z + 0.5,
          loadedChunkData
        ); // Check center of voxel
        if (isSolidVoxel(voxelType)) {
          const voxelAABB = getVoxelAABB(x, y, z);
          // Double check overlap before adding (though the loop bounds should handle most cases)
          if (doesAABBOverlap(aabb, voxelAABB)) {
            collisions.push(voxelAABB);
          }
        }
      }
    }
  }
  return collisions;
}

// --- Movement Calculation Helpers ---

function calculateDesiredVelocity(
  pressedKeys: Set<string>,
  cameraYaw: number,
  currentVelocity: vec3 // Pass current velocity if needed for complex movement
): vec3 {
  const desiredVelocity = vec3.create();
  const cameraUp = vec3.fromValues(0, 1, 0);

  const forward = vec3.create();
  forward[0] = Math.sin(cameraYaw);
  forward[2] = Math.cos(cameraYaw);
  vec3.normalize(forward, forward);

  const right = vec3.create();
  vec3.cross(right, forward, cameraUp);
  vec3.normalize(right, right);

  if (pressedKeys.has("KeyW")) {
    vec3.add(desiredVelocity, desiredVelocity, forward);
  }
  if (pressedKeys.has("KeyS")) {
    vec3.subtract(desiredVelocity, desiredVelocity, forward);
  }
  if (pressedKeys.has("KeyA")) {
    vec3.subtract(desiredVelocity, desiredVelocity, right);
  }
  if (pressedKeys.has("KeyD")) {
    vec3.add(desiredVelocity, desiredVelocity, right);
  }

  if (desiredVelocity[0] !== 0 || desiredVelocity[2] !== 0) {
    vec3.normalize(desiredVelocity, desiredVelocity);
    vec3.scale(desiredVelocity, desiredVelocity, MOVE_SPEED);
  }

  return desiredVelocity;
}

function applyGravity(velocity: vec3, deltaTimeSeconds: number) {
  velocity[1] += GRAVITY * deltaTimeSeconds;
}

function handleJump(velocity: vec3, isGrounded: boolean): boolean {
  if (isGrounded) {
    velocity[1] = JUMP_VELOCITY;
    return false; // No longer grounded
  }
  return isGrounded; // Remain not grounded if already in air
}

// --- AABB Collision Resolution ---

/**
 * Calculates the maximum distance an AABB can move along a single axis before colliding
 * with any of the provided potential colliders.
 *
 * @param currentAABB The starting AABB.
 * @param displacement The desired movement along the axis.
 * @param axisIndex 0 for X, 1 for Y, 2 for Z.
 * @param potentialColliders An array of AABBs representing potential obstacles.
 * @returns The maximum allowed displacement along the axis (can be less than input displacement if collision occurs).
 */
function resolveCollisionsAxis(
  currentAABB: AABB,
  displacement: number,
  axisIndex: number,
  potentialColliders: AABB[]
): number {
  if (displacement === 0 || potentialColliders.length === 0) {
    return displacement;
  }

  let maxDisplacement = displacement;
  const movingPositive = displacement > 0;

  // Create a swept AABB for this axis movement only
  const sweptAxisAABB = expandAABB(
    currentAABB,
    axisIndex === 0 ? displacement : 0,
    axisIndex === 1 ? displacement : 0,
    axisIndex === 2 ? displacement : 0
  );

  for (const collider of potentialColliders) {
    // Basic overlap check first on the swept AABB
    if (!doesAABBOverlap(sweptAxisAABB, collider)) {
      continue;
    }

    // Check if already overlapping slightly (or exactly touching) on other axes
    let overlapsOnOtherAxes = true;
    for (let i = 0; i < 3; i++) {
      if (i === axisIndex) continue;
      if (
        currentAABB.max[i] <= collider.min[i] ||
        currentAABB.min[i] >= collider.max[i]
      ) {
        overlapsOnOtherAxes = false;
        break;
      }
    }
    if (!overlapsOnOtherAxes) {
      continue;
    }

    // Calculate exact collision point on this axis
    if (movingPositive) {
      const distanceToCollision =
        collider.min[axisIndex] - currentAABB.max[axisIndex];
      if (distanceToCollision >= 0 && distanceToCollision < maxDisplacement) {
        maxDisplacement = distanceToCollision - COLLISION_EPSILON; // Move just before touching
      }
    } else {
      // Moving negative
      const distanceToCollision =
        collider.max[axisIndex] - currentAABB.min[axisIndex];
      if (distanceToCollision <= 0 && distanceToCollision > maxDisplacement) {
        maxDisplacement = distanceToCollision + COLLISION_EPSILON; // Move just before touching
      }
    }
    // Ensure we don't accidentally reverse direction due to epsilon adjustments
    if (
      (movingPositive && maxDisplacement < 0) ||
      (!movingPositive && maxDisplacement > 0)
    ) {
      maxDisplacement = 0;
    }
  }

  // Prevent minuscule movements due to floating point errors
  if (Math.abs(maxDisplacement) < COLLISION_EPSILON * 2) {
    return 0;
  }

  return maxDisplacement;
}

// --- Physics Update Function (To be refactored for AABB) ---
export function updatePhysics(
  playerState: PlayerState,
  pressedKeys: Set<string>,
  cameraYaw: number, // Needed for movement direction
  deltaTimeSeconds: number,
  loadedChunkData: Map<string, Uint8Array> // Pass chunk data map
): PlayerState {
  const { position, velocity } = playerState;
  let isGrounded = playerState.isGrounded;

  // 1. Calculate Desired Velocity based on input
  const desiredVelocity = calculateDesiredVelocity(
    pressedKeys,
    cameraYaw,
    velocity
  );

  // 2. Apply Gravity
  if (!ENABLE_FLYING_MODE) {
    applyGravity(velocity, deltaTimeSeconds);

    if (pressedKeys.has("Space")) {
      isGrounded = handleJump(velocity, isGrounded);
    }
  } else {
    velocity[1] = 0;
    if (pressedKeys.has("Space")) {
      velocity[1] = JUMP_VELOCITY;
    }
    if (pressedKeys.has("ShiftLeft")) {
      velocity[1] = -JUMP_VELOCITY;
    }
  }

  // 4. Calculate displacement for this frame
  // Horizontal movement is based purely on intent for this frame
  // Vertical movement includes gravity/jump velocity accumulated
  const dx = desiredVelocity[0] * deltaTimeSeconds;
  const dy = velocity[1] * deltaTimeSeconds;
  const dz = desiredVelocity[2] * deltaTimeSeconds;

  // 5. Resolve Collisions and Apply Movement using AABB

  const currentAABB = getPlayerAABB(position);

  // 5.1 Broad Phase: Find potential colliders based on *total* potential movement
  const sweptAABB = expandAABB(currentAABB, dx, dy, dz);
  const potentialCollisions = getPotentialVoxelCollisions(
    sweptAABB,
    loadedChunkData
  );

  // 5.2 Resolve Y-axis
  const resolvedDy = resolveCollisionsAxis(
    currentAABB,
    dy,
    1,
    potentialCollisions
  );
  position[1] += resolvedDy;
  const collidedY = Math.abs(resolvedDy - dy) > COLLISION_EPSILON;
  if (collidedY) {
    if (dy < 0) {
      // Collided while moving down
      isGrounded = true;
    }

    velocity[1] = 0; // Stop vertical velocity on collision
  } else {
    isGrounded = false; // Not grounded if moved freely vertically
  }
  // Update AABB after Y movement before resolving X/Z
  let intermediateAABB = getPlayerAABB(position);

  // 5.3 Resolve X-axis
  const resolvedDx = resolveCollisionsAxis(
    intermediateAABB,
    dx,
    0,
    potentialCollisions
  );
  position[0] += resolvedDx;
  const collidedX = Math.abs(resolvedDx - dx) > COLLISION_EPSILON;
  if (collidedX) {
    // TODO: Implement Stepping check here
    // If stepping fails or is not applicable:
    velocity[0] = 0; // Stop X velocity on collision
  }
  // Update AABB again after X movement
  intermediateAABB = getPlayerAABB(position);

  // 5.4 Resolve Z-axis
  const resolvedDz = resolveCollisionsAxis(
    intermediateAABB,
    dz,
    2,
    potentialCollisions
  );
  position[2] += resolvedDz;
  const collidedZ = Math.abs(resolvedDz - dz) > COLLISION_EPSILON;
  if (collidedZ) {
    // TODO: Implement Stepping check here
    // If stepping fails or is not applicable:
    velocity[2] = 0; // Stop Z velocity on collision
  }

  if (isGrounded) {
    velocity[1] = 0;
  }

  // Return the updated state
  return PlayerState.fromValues(position, velocity, isGrounded);
}
