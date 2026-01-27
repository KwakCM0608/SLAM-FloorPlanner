# slam_floorplan

Pipeline to generate a 2D floorplan from a phone video using COLMAP (SfM/MVS) + post-processing.

## Steps (high level)
1. Extract frames from video
2. COLMAP sparse reconstruction
3. COLMAP dense (patch_match_stereo + stereo_fusion) -> fused.ply
4. Generate 2D floorplan from fused.ply

## Notes
- Large artifacts (work/, output/, *.ply, *.db) are ignored via .gitignore.
