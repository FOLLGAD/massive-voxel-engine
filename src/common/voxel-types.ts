// 0 is reserved for Air (empty space)
export enum VoxelType {
  AIR = 0,
  STONE = 1,
  GRASS = 2, // Let's add another for variation
  DIRT = 3,
  SAND = 4,
  STAR = 5,
  WATER = 6,
  LAVA = 7,
  GLASS = 8,
  IRON = 9,
  GOLD = 10,
  DIAMOND = 11,
  EMERALD = 12,
  LAPIS_LAZULI = 13,
  REDSTONE = 14,
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
    case VoxelType.WATER:
      return [0.0, 0.0, 1.0]; // Blue for WATER
    case VoxelType.LAVA:
      return [1.0, 0.0, 0.0]; // Red for LAVA
    case VoxelType.GLASS:
      return [0.8, 0.8, 0.8]; // Gray for GLASS
    case VoxelType.IRON:
      return [0.5, 0.5, 0.5]; // Gray for IRON
    case VoxelType.GOLD:
      return [1.0, 0.8, 0.0]; // Gold for GOLD
    case VoxelType.DIAMOND:
      return [0.0, 0.0, 1.0]; // Blue for DIAMOND
    case VoxelType.EMERALD:
      return [0.0, 1.0, 0.0]; // Green for EMERALD
    case VoxelType.LAPIS_LAZULI:
      return [0.0, 0.0, 0.8]; // Blue for LAPIS_LAZULI
    case VoxelType.REDSTONE:
      return [1.0, 0.0, 0.0]; // Red for REDSTONE

    default:
      return [0.0, 0.0, 0.0]; // Black for AIR or unknown
  }
}
