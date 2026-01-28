#!/usr/bin/env python3
import argparse, os, json
import numpy as np
import cv2
import open3d as o3d

def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _rot_from_a_to_b(a, b):
    # Rodrigues rotation taking vector a to b
    a = _unit(a); b = _unit(b)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-10:
        # parallel or anti-parallel
        if c > 0:
            return np.eye(3)
        # 180 deg: pick any orthogonal axis
        axis = _unit(np.cross(a, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-6:
            axis = _unit(np.cross(a, np.array([0.0, 1.0, 0.0])))
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        return np.eye(3) + 2 * (K @ K)  # since sin(pi)=0, 1-cos(pi)=2
    axis = v / s
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    R = np.eye(3) + K * s + (K @ K) * (1 - c)
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='Input PLY (fused.ply)')
    ap.add_argument(
    '--out',
    default=os.path.expanduser('~/slam_floorplan/output'),
    help='Output directory (default: ~/slam_floorplan/output)'
    )
 
    ap.add_argument('--res', type=float, default=0.03, help='Grid resolution in meters (smaller=more detail)')
    ap.add_argument('--voxel', type=float, default=0.03, help='Voxel downsample for plane finding')
    ap.add_argument('--nb_neighbors', type=int, default=25, help='Outlier removal neighbors')
    ap.add_argument('--std_ratio', type=float, default=2.0, help='Outlier removal std ratio')
    ap.add_argument('--plane_dist', type=float, default=0.03, help='RANSAC plane distance threshold (m)')
    ap.add_argument('--ransac_n', type=int, default=3)
    ap.add_argument('--ransac_iter', type=int, default=4000)

    # floor band selection
    ap.add_argument('--floor_band', type=float, default=0.12, help='Half band around floor z (m). Uses z_floor±band')
    ap.add_argument('--max_floor_z', type=float, default=0.50, help='Ignore candidate floor above this after alignment (m)')

    # density -> binary
    ap.add_argument('--thr_mode', choices=['otsu', 'percentile'], default='percentile')
    ap.add_argument('--thr_percentile', type=float, default=92.0, help='Percentile for density threshold (if percentile mode)')
    ap.add_argument('--min_area', type=int, default=2000, help='Remove components smaller than this (pixels)')
    ap.add_argument('--keep_largest', action='store_true', help='Keep only the largest connected component')

    # morphology
    ap.add_argument('--close_k', type=int, default=11)
    ap.add_argument('--close_iter', type=int, default=2)
    ap.add_argument('--open_k', type=int, default=5)
    ap.add_argument('--open_iter', type=int, default=1)
    ap.add_argument('--dilate_k', type=int, default=3)
    ap.add_argument('--dilate_iter', type=int, default=1)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ply = os.path.expanduser(args.model)
    if not os.path.exists(ply):
        raise FileNotFoundError(ply)

    print("[1] Loading:", ply)
    pcd = o3d.io.read_point_cloud(ply)
    if len(pcd.points) == 0:
        raise RuntimeError("Empty point cloud")

    # downsample + denoise (for stable plane)
    p = pcd.voxel_down_sample(args.voxel)
    p, _ = p.remove_statistical_outlier(nb_neighbors=args.nb_neighbors, std_ratio=args.std_ratio)

    print("[2] Plane segmentation (floor) ...")
    plane_model, inliers = p.segment_plane(distance_threshold=args.plane_dist,
                                           ransac_n=args.ransac_n,
                                           num_iterations=args.ransac_iter)
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n = _unit(n)

    # Make normal point "up" (positive Z after rotation target)
    # We'll rotate plane normal to +Z
    R = _rot_from_a_to_b(n, np.array([0.0, 0.0, 1.0]))

    # Apply to full-res points (not downsampled)
    pts = np.asarray(pcd.points, dtype=np.float64)
    pts_aligned = (R @ pts.T).T

    # After rotation, the floor plane becomes z = const.
    # Estimate floor z from plane inliers (use full aligned pts for robustness):
    inlier_pts = np.asarray(p.select_by_index(inliers).points, dtype=np.float64)
    inlier_aligned = (R @ inlier_pts.T).T
    z_floor = float(np.median(inlier_aligned[:, 2]))

    # Sometimes the "big plane" can be desk/wall; clamp using max_floor_z heuristic:
    # We'll search a bit downward if z_floor is suspiciously high.
    if z_floor > args.max_floor_z:
        # fallback: take the lowest strong mode of z histogram
        z = pts_aligned[:, 2]
        z_clip = z[np.isfinite(z)]
        hist, edges = np.histogram(z_clip, bins=200)
        idx = int(np.argmax(hist[:120]))  # focus on lower part
        z_floor = float((edges[idx] + edges[idx+1]) * 0.5)

    print(f"[3] Estimated floor z ≈ {z_floor:.3f} m (aligned frame)")

    # Slice band around floor (keep near-floor structure)
    band = args.floor_band
    m = (pts_aligned[:, 2] >= z_floor - band) & (pts_aligned[:, 2] <= z_floor + band)
    pts_floor = pts_aligned[m]
    if pts_floor.shape[0] < 1000:
        raise RuntimeError(f"Too few points in floor band: {pts_floor.shape[0]}. "
                           f"Try increasing --floor_band (e.g. 0.2~0.35)")

    # Project to XY density grid
    xy = pts_floor[:, :2]
    mn = xy.min(axis=0)
    xy0 = xy - mn
    wh = np.ceil(xy0.max(axis=0) / args.res).astype(int) + 1
    W, H = int(wh[0]), int(wh[1])

    # Density accumulation
    ij = np.floor(xy0 / args.res).astype(np.int32)
    den = np.zeros((H, W), dtype=np.uint16)
    # safe indexing
    ij[:, 0] = np.clip(ij[:, 0], 0, W-1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H-1)
    np.add.at(den, (ij[:, 1], ij[:, 0]), 1)

    # Normalize density to 0..255 for visualization
    den_vis = den.astype(np.float32)
    den_vis = den_vis / (np.percentile(den_vis[den_vis > 0], 99.5) + 1e-6) * 255.0
    den_vis = np.clip(den_vis, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(args.out, "floorplan_density.png"), den_vis)

    # Threshold
    if args.thr_mode == "otsu":
        thr_src = den_vis
        _, bw = cv2.threshold(thr_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        v = float(np.percentile(den_vis[den_vis > 0], args.thr_percentile)) if np.any(den_vis > 0) else 255.0
        _, bw = cv2.threshold(den_vis, int(v), 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.join(args.out, "floorplan_thr.png"), bw)

    # Morphology cleanup (gentle by default)
    def k(n): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    img = bw.copy()
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k(args.close_k), iterations=args.close_iter)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN,  k(args.open_k),  iterations=args.open_iter)
    img = cv2.dilate(img, k(args.dilate_k), iterations=args.dilate_iter)

    # Remove small components + optional keep largest
    num, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    keep = []
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= args.min_area:
            keep.append(i)

    if len(keep) == 0:
        raise RuntimeError("All components removed. Try lower --min_area or lower threshold (percentile).")

    if args.keep_largest:
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in keep]
        keep = [max(areas, key=lambda x: x[1])[0]]

    clean = np.zeros_like(img)
    for i in keep:
        clean[labels == i] = 255

    cv2.imwrite(os.path.join(args.out, "floorplan_clean.png"), clean)

    # Debug / metadata
    meta = {
        "input_ply": ply,
        "res_m": args.res,
        "voxel_m": args.voxel,
        "plane_model": [float(a), float(b), float(c), float(d)],
        "plane_normal_unit": n.tolist(),
        "z_floor_aligned_m": z_floor,
        "floor_band_m": band,
        "thr_mode": args.thr_mode,
        "thr_percentile": args.thr_percentile,
        "origin_xy_m": mn.tolist(),
        "grid_W": W,
        "grid_H": H
    }
    with open(os.path.join(args.out, "floorplan_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Also save aligned ply (optional, useful for check)
    pcd_aligned = o3d.geometry.PointCloud()
    pcd_aligned.points = o3d.utility.Vector3dVector(pts_aligned)
    o3d.io.write_point_cloud(os.path.join(args.out, "fused_aligned.ply"), pcd_aligned)

    print("[OK] Saved:")
    print(" - floorplan_density.png")
    print(" - floorplan_thr.png")
    print(" - floorplan_clean.png")
    print(" - fused_aligned.ply")
    print(" - floorplan_meta.json")
    print("Output dir:", args.out)

if __name__ == "__main__":
    main()
