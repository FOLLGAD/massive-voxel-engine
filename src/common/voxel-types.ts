// 0 is reserved for Air (empty space)
export enum VoxelType {
  AIR = 0,
  STONE = 1,
  GRASS = 2, // Let's add another for variation
  DIRT = 3,
  SAND = 4,
  STAR = 5,
}

export function isVoxelSolid(voxelType: VoxelType): boolean {
  return voxelType !== VoxelType.AIR;
}

export function getVoxelColor(voxelType: VoxelType): [number, number, number] {
  switch (voxelType) {
    case VoxelType.STONE:
      return [0.5, 0.5, 0.5]; // Gray
    case VoxelType.GRASS:
      return [0.2, 0.6, 0.2]; // Green
    case VoxelType.DIRT:
      return [0.6, 0.4, 0.2]; // Brown
    case VoxelType.AIR:
      return [0.0, 0.0, 0.0]; // Black for AIR or unknown
    case VoxelType.SAND:
      return [1.0, 1.0, 0.0]; // Yellow for SAND
    case VoxelType.STAR:
      return [0.8, 0.4, 0.2]; // Red for STAR
    default:
      return [0.0, 0.0, 0.0]; // Black for AIR or unknown
  }
}
