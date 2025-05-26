import { vec3, vec4 } from "gl-matrix";
import { CHUNK_SIZE_X, CHUNK_SIZE_Y, CHUNK_SIZE_Z } from "./config";
import type { AABB } from "./aabb";
import type { ChunkGeometryInfo } from "./chunk-manager";
import type { Plane } from "./renderer";
import { getChunkKey } from "./chunk";

// Octree node size levels (in chunks)
// Level 0: 1x1x1 chunks (individual chunks)
// Level 1: 2x2x2 chunks
// Level 2: 4x4x4 chunks
// Level 3: 8x8x8 chunks
// etc.
const MAX_OCTREE_LEVELS = 4; // Adjust based on world size

interface OctreeNode {
	level: number;
	position: vec3; // Position in octree coordinates (not world coordinates)
	aabb: AABB;
	children: (OctreeNode | null)[];
	chunks: Map<string, ChunkGeometryInfo>; // Only for leaf nodes
	visibilityBits: number; // Aggregated visibility for non-leaf nodes
	isEmpty: boolean; // True if no chunks exist in this subtree
}

export class HierarchicalOctree {
	private root: OctreeNode | null = null;
	private chunkToNode: Map<string, OctreeNode> = new Map();

	// Convert chunk position to octree position at given level
	private chunkPosToOctreePos(chunkPos: vec3, level: number): vec3 {
		const scale = 2 ** level;
		return vec3.fromValues(
			Math.floor(chunkPos[0] / scale),
			Math.floor(chunkPos[1] / scale),
			Math.floor(chunkPos[2] / scale),
		);
	}

	// Convert octree position to world-space AABB
	private octreePosToAABB(octreePos: vec3, level: number): AABB {
		const scale = 2 ** level;
		const sizeInChunks = scale;

		const minWorld = vec3.fromValues(
			octreePos[0] * sizeInChunks * CHUNK_SIZE_X,
			octreePos[1] * sizeInChunks * CHUNK_SIZE_Y,
			octreePos[2] * sizeInChunks * CHUNK_SIZE_Z,
		);

		const maxWorld = vec3.fromValues(
			(octreePos[0] + 1) * sizeInChunks * CHUNK_SIZE_X,
			(octreePos[1] + 1) * sizeInChunks * CHUNK_SIZE_Y,
			(octreePos[2] + 1) * sizeInChunks * CHUNK_SIZE_Z,
		);

		return { min: minWorld, max: maxWorld };
	}

	// Get child index (0-7) for a position within parent
	private getChildIndex(childOctreePos: vec3, parentOctreePos: vec3): number {
		const dx = childOctreePos[0] - parentOctreePos[0] * 2;
		const dy = childOctreePos[1] - parentOctreePos[1] * 2;
		const dz = childOctreePos[2] - parentOctreePos[2] * 2;
		return dx + dy * 2 + dz * 4;
	}

	// Create a new octree node
	private createNode(level: number, position: vec3): OctreeNode {
		return {
			level,
			position: vec3.clone(position),
			aabb: this.octreePosToAABB(position, level),
			children: new Array(8).fill(null),
			chunks: level === 0 ? new Map() : new Map(),
			visibilityBits: 0,
			isEmpty: true,
		};
	}

	// Add a chunk to the octree
	public addChunk(chunkInfo: ChunkGeometryInfo): void {
		const chunkKey = getChunkKey(chunkInfo.position);

		// Remove from old position if it exists
		this.removeChunk(chunkKey);

		// Find or create the leaf node for this chunk
		const leafPos = this.chunkPosToOctreePos(chunkInfo.position, 0);
		const leafNode = this.ensureNodeExists(leafPos, 0);

		// Add chunk to leaf node
		leafNode.chunks.set(chunkKey, chunkInfo);
		leafNode.isEmpty = false;
		leafNode.visibilityBits = chunkInfo.visibilityBits;

		// Store mapping
		this.chunkToNode.set(chunkKey, leafNode);

		// Update parent nodes
		this.updateParentNodes(leafNode);
	}

	// Ensure a node exists at the given position and level
	private ensureNodeExists(octreePos: vec3, level: number): OctreeNode {
		if (!this.root) {
			// Create root at highest level that contains this position
			const rootLevel = MAX_OCTREE_LEVELS - 1;
			const rootPos = this.chunkPosToOctreePos(octreePos, rootLevel);
			this.root = this.createNode(rootLevel, rootPos);
		}

		// Navigate down from root, creating nodes as needed
		let current = this.root;
		let currentLevel = this.root.level;

		while (currentLevel > level) {
			const childLevel = currentLevel - 1;
			const childPos = this.chunkPosToOctreePos(octreePos, childLevel);
			const childIndex = this.getChildIndex(childPos, current.position);

			if (!current.children[childIndex]) {
				current.children[childIndex] = this.createNode(childLevel, childPos);
			}

			// biome-ignore lint/style/noNonNullAssertion: <explanation>
			current = current.children[childIndex]!;
			currentLevel = childLevel;
		}

		return current;
	}

	// Update visibility bits and empty status up the tree
	private updateParentNodes(node: OctreeNode): void {
		let current = node;
		const path: OctreeNode[] = [current];

		// Build path to root
		while (current.level < (this.root?.level ?? 0)) {
			const parentLevel = current.level + 1;
			const parentPos = this.chunkPosToOctreePos(current.position, parentLevel);
			const parent = this.ensureNodeExists(parentPos, parentLevel);
			path.push(parent);
			current = parent;
		}

		// Update from bottom to top
		for (let i = path.length - 1; i >= 0; i--) {
			const node = path[i];
			if (node.level === 0) continue; // Skip leaf nodes

			// Aggregate from children
			let aggregatedVisibility = 0;
			let hasNonEmptyChild = false;

			for (const child of node.children) {
				if (child && !child.isEmpty) {
					aggregatedVisibility |= child.visibilityBits;
					hasNonEmptyChild = true;
				}
			}

			node.visibilityBits = aggregatedVisibility;
			node.isEmpty = !hasNonEmptyChild;
		}
	}

	// Remove a chunk from the octree
	public removeChunk(chunkKey: string): void {
		const node = this.chunkToNode.get(chunkKey);
		if (!node) return;

		node.chunks.delete(chunkKey);
		this.chunkToNode.delete(chunkKey);

		// Update empty status
		if (node.chunks.size === 0) {
			node.isEmpty = true;
			node.visibilityBits = 0;
			this.updateParentNodes(node);

			// TODO: Consider pruning empty branches to save memory
		}
	}

	// Cull chunks hierarchically
	public cullChunksHierarchical(
		frustumPlanes: Plane[],
		cameraPosition: vec3,
		intersectFrustumAABB: (planes: Plane[], aabb: AABB) => boolean,
	): ChunkGeometryInfo[] {
		if (!this.root || this.root.isEmpty) {
			return [];
		}

		const visibleChunks: ChunkGeometryInfo[] = [];
		const nodesToCheck: OctreeNode[] = [this.root];

		while (nodesToCheck.length > 0) {
			// biome-ignore lint/style/noNonNullAssertion: <explanation>
			const node = nodesToCheck.pop()!;

			// Skip empty nodes
			if (node.isEmpty) continue;

			// Frustum cull the node's AABB
			if (!intersectFrustumAABB(frustumPlanes, node.aabb)) {
				// Entire subtree is culled
				continue;
			}

			// If leaf node, add its chunks
			if (node.level === 0) {
				for (const chunk of node.chunks.values()) {
					visibleChunks.push(chunk);
				}
			} else {
				// Add non-empty children for further testing
				for (const child of node.children) {
					if (child && !child.isEmpty) {
						nodesToCheck.push(child);
					}
				}
			}
		}

		return visibleChunks;
	}

	// Get statistics about the octree
	public getStats(): {
		totalNodes: number;
		nodesByLevel: Map<number, number>;
		totalChunks: number;
		maxDepth: number;
	} {
		const stats = {
			totalNodes: 0,
			nodesByLevel: new Map<number, number>(),
			totalChunks: this.chunkToNode.size,
			maxDepth: 0,
		};

		if (!this.root) return stats;

		const traverse = (node: OctreeNode) => {
			stats.totalNodes++;
			stats.nodesByLevel.set(
				node.level,
				(stats.nodesByLevel.get(node.level) ?? 0) + 1,
			);
			stats.maxDepth = Math.max(stats.maxDepth, node.level);

			for (const child of node.children) {
				if (child) traverse(child);
			}
		};

		traverse(this.root);
		return stats;
	}

	// Clear the entire octree
	public clear(): void {
		this.root = null;
		this.chunkToNode.clear();
	}
}
