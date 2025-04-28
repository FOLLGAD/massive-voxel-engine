import { vec3, vec4, type mat4 } from "gl-matrix";
import type { Plane } from "./renderer";

// --- Frustum Culling Helpers (Static) ---
export function extractFrustumPlanes(mat: mat4): Plane[] {
  const planes: Plane[] = [
    vec4.create(),
    vec4.create(),
    vec4.create(),
    vec4.create(),
    vec4.create(),
    vec4.create(),
  ];
  const m = mat;
  const get = (i: number, j: number) => m[i * 4 + j];

  for (let i = 4; i--; ) planes[0][i] = get(i, 3) + get(i, 0); // Left
  for (let i = 4; i--; ) planes[1][i] = get(i, 3) - get(i, 0); // Right
  for (let i = 4; i--; ) planes[2][i] = get(i, 3) + get(i, 1); // Bottom
  for (let i = 4; i--; ) planes[3][i] = get(i, 3) - get(i, 1); // Top
  for (let i = 4; i--; ) planes[4][i] = get(i, 3) + get(i, 2); // Near
  for (let i = 4; i--; ) planes[5][i] = get(i, 3) - get(i, 2); // Far

  for (const plane of planes) {
    const invLength = 1.0 / vec3.length([plane[0], plane[1], plane[2]]);
    vec4.scale(plane, plane, invLength);
  }
  return planes;
}
