# massive-voxel-engine

<img width="1481" alt="Screenshot 2025-04-21 at 11 14 16" src="https://github.com/user-attachments/assets/45abf027-e641-4030-933b-eb5bc2261756" />

My experiments with making a voxel engine.

Built on top of WebGPU.

Currently generates terrain with extensive cave systems.

## Goals

- Cool terrain features
- Highly performant – can run on any consumer laptop at 60fps
- portals!!!
-

## Todo

- culling on GPU + indirect drawing
- occlusion culling (GPU)
- geometry batching
- play around with different rendering styles

  - blue noise
  - pixellation

- occlusion culling
  - for each block in a chunk, do a flood fill in that chunk.
  - if a flood fill connects two sides of the outer bound of the chunk, mark those two sides as "visually connected"
    - each handled block goes into a "handledblockmap", and is skipped in upcoming flood fill attempts.
  - this data is stored in the chunk info, and updated when a chunk is updated
  - the chunk data visbility info can then be queried to decide whether a specific chunk should be rendered:
    - when rendering, start with queue A={player's chunk}
      - for each neighboring chunk B:
        - if going there does not require going back in the opposite of the player's view, and
        - the chunk's visibility data says B can be viewed from A (from the correct faces)
        - then, push it to the queue and mark it as non-culled.
  - then, either:
    - just render all the non-culled chunks
    - (BONKERS IDEA:) or, do a first render pass replacing each non-culled chunk with a 6-sided cube where each face is rendered only if it's walkable according to the visibility matrix.
      - do a occlusion query and then render the chunks that are queried as visible
