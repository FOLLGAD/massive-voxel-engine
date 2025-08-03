import { vec3 } from "gl-matrix";
import { PLAYER_HALF_WIDTH } from "./physics";
import { PLAYER_HEIGHT } from "./physics";

export type AABB = {
  min: vec3;
  max: vec3;
};

export function createAABB(min: vec3, max: vec3): AABB {
  return { min, max };
}

export function doesAABBOverlap(a: AABB, b: AABB): boolean {
  return (
    a.min[0] < b.max[0] &&
    a.max[0] > b.min[0] &&
    a.min[1] < b.max[1] &&
    a.max[1] > b.min[1] &&
    a.min[2] < b.max[2] &&
    a.max[2] > b.min[2]
  );
}

export function getPlayerAABB(cameraPos: vec3): { min: vec3; max: vec3 } {
  const minY = cameraPos[1] - PLAYER_HEIGHT;
  const maxY = cameraPos[1];
  return {
    min: vec3.fromValues(
      cameraPos[0] - PLAYER_HALF_WIDTH,
      minY,
      cameraPos[2] - PLAYER_HALF_WIDTH
    ),
    max: vec3.fromValues(
      cameraPos[0] + PLAYER_HALF_WIDTH,
      maxY,
      cameraPos[2] + PLAYER_HALF_WIDTH
    ),
  };
}

export function expandAABB(aabb: AABB, x: number, y: number, z: number): AABB {
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

