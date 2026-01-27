#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

# Open3D: core만 사용 (open3d.ml 불필요)
import open3d as o3d

# OpenCV for fast morphology + connected components
import cv2


def rot_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return rotation matrix that rotates vector a to vector b."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3, dtype=np.float64)
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]], dtype=np.float64)
    # Rodrigues
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s * s + 1e-12))
    return R


def find_floor_plane_ransac(pcd: o3d.geometry.PointCloud,
                           dist_thresh: float,
                           iters: int,
                           tries: int = 6):
    """
    Find a good floor-like plane using multiple RANSAC runs.
    Heuristic: prefer planes close to horizontal (|nz| high) and low in Z.
    """
    rest = pcd
    best = None
    for _ in range(tries):
        if len(rest.points) < 800:
            break
        plane_model, inliers = rest.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=3,
            num_iterations=iters
        )
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=np.float64)
        n_norm = np.linalg.norm(n) + 1e-12
        n = n / n_norm
        # plane points stats
        plane_cloud = rest.select_by_index(inliers)
        pts = np.asarray(plane_cloud.points)
        z_mean = float(np.mean(pts[:, 2]))
        horiz = abs(float(n[2]))  # close to 1 means horizontal
        # score: favor horizontal + lower z
        score = horiz * 2.0 + (-z_mean) * 0.25

        if best is None or score > best["score"]:
            best = {"plane": plane_model, "n": n, "z_mean": z_mean, "score": score}

        # remove this plane and keep searching other large planes
        rest = rest.select_by_index(inliers, invert=True)

    if best is None:
        raise RuntimeError("Failed to find a plane (floor) with RANSAC.")
    return best


def rasterize_xy_fast(xy: np.ndarray, res: float, margin: float):
    """Rasterize XY points into a binary occupancy image using bincount (fast)."""
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)

    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    W = int(np.ceil((xmax - xmin) / res))
    H = int(np.ceil((ymax - ymin) / res))
    W = max(W, 200)
    H = max(H, 200)

    ix = ((xy[:, 0] - xmin) / res).astype(np.int32)
    iy = ((xy[:, 1] - ymin) / res).astype(np.int32)

    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix = ix[valid]
    iy = iy[valid]

    # Linear index and bincount
    lin = iy.astype(np.int64) * np.int64(W) + ix.astype(np.int64)
    counts = np.bincount(lin, minlength=H * W)

    img = (counts.reshape(H, W) > 0).astype(np.uint8) * 255
    # Flip to conventional image coordinates (y down)
    img = np.flipud(img)
    return img, (xmin, ymin, xmax, ymax, W, H)


def remove_small_components(binary_img_255: np.ndarray, min_area: int):
    """Keep only connected components with area >= min_area. Input 0/255."""
    if min_area <= 0:
        return binary_img_255
    bw = (binary_img_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(binary_img_255)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 255
    return out


def hough_lines_overlay(binary_img_255: np.ndarray,
                        min_line_length: int,
                        max_line_gap: int,
                        manhattan: bool = True):
    """
    Optional: extract lines from occupancy to make it look more like a floor plan.
    Returns an RGB image with lines drawn.
    """
    # Edges
    edges = cv2.Canny(binary_img_255, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=60,
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    out = cv2.cvtColor(binary_img_255, cv2.COLOR_GRAY2BGR)
    if lines is None:
        return out

    # Manhattan snap: keep near-horizontal/vertical lines
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx = x2 - x1
        dy = y2 - y1
        if manhattan:
            ang = np.degrees(np.arctan2(dy, dx))
            ang = (ang + 180.0) % 180.0
            # accept near 0 or 90
            if not (ang < 15.0 or abs(ang - 90.0) < 15.0 or abs(ang - 180.0) < 15.0):
                continue
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red lines
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, help="Input point cloud (.ply), e.g., dense fused.ply")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--res", type=float, default=0.05, help="Grid resolution (m/pixel)")
    ap.add_argument("--voxel", type=float, default=0.03, help="Voxel downsample size (m)")

    # Floor finding / alignment
    ap.add_argument("--plane_dist", type=float, default=0.06, help="RANSAC plane distance threshold (m)")
    ap.add_argument("--plane_iters", type=int, default=3000, help="RANSAC iterations per try")
    ap.add_argument("--plane_tries", type=int, default=6, help="Number of plane tries")

    # Height slice relative to floor z0
    ap.add_argument("--zmin", type=float, default=0.20, help="Keep points above floor by this (m)")
    ap.add_argument("--zmax", type=float, default=1.60, help="Keep points above floor up to this (m)")

    # Cleaning
    ap.add_argument("--margin", type=float, default=0.50, help="Map margin around points (m)")
    ap.add_argument("--close_k", type=int, default=11, help="Morph close kernel size (odd recommended)")
    ap.add_argument("--close_iter", type=int, default=3, help="Morph close iterations")
    ap.add_argument("--open_k", type=int, default=7, help="Morph open kernel size (odd recommended)")
    ap.add_argument("--open_iter", type=int, default=2, help="Morph open iterations")
    ap.add_argument("--dilate_k", type=int, default=3, help="Initial dilate kernel size")
    ap.add_argument("--dilate_iter", type=int, default=2, help="Initial dilate iterations")
    ap.add_argument("--min_area", type=int, default=800, help="Remove components smaller than this area (px)")

    # Optional speed/robustness filters
    ap.add_argument("--stat_outlier", action="store_true",
                    help="Apply statistical outlier removal (slower but cleaner)")
    ap.add_argument("--stat_nb", type=int, default=20, help="SOR: nb_neighbors")
    ap.add_argument("--stat_std", type=float, default=2.0, help="SOR: std_ratio")

    # Optional line overlay
    ap.add_argument("--draw_lines", action="store_true", help="Also output a Hough-lines overlay image")
    ap.add_argument("--min_line_len", type=int, default=60, help="Hough min line length (px)")
    ap.add_argument("--max_line_gap", type=int, default=10, help="Hough max line gap (px)")
    ap.add_argument("--no_manhattan", action="store_true", help="Disable Manhattan (0/90 deg) preference in lines")

    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    pcd = o3d.io.read_point_cloud(args.ply)
    if pcd.is_empty():
        raise RuntimeError(f"Point cloud is empty: {args.ply}")
    print("Loaded points:", len(pcd.points))

    # --- Downsample ---
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
        print("After voxel downsample:", len(pcd.points))

    # --- Optional outlier removal (slower, cleaner) ---
    if args.stat_outlier:
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=args.stat_nb, std_ratio=args.stat_std)
        print("After statistical outlier removal:", len(pcd.points))

    # --- Find floor plane and align to Z-up ---
    best = find_floor_plane_ransac(pcd, dist_thresh=args.plane_dist, iters=args.plane_iters, tries=args.plane_tries)
    n = best["n"].copy()
    if n[2] < 0:
        n = -n
    R = rot_from_a_to_b(n, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    pts = np.asarray(pcd.points, dtype=np.float64)
    pts_aligned = (R @ pts.T).T

    # Robust floor height estimate (low percentile)
    z = pts_aligned[:, 2]
    z0 = float(np.percentile(z, 5))
    print("Estimated floor z0:", z0)

    # --- Height slice for wall candidates ---
    keep = (z > z0 + args.zmin) & (z < z0 + args.zmax)
    xy = pts_aligned[keep, :2]
    print("Kept points for 2D:", xy.shape[0])

    if xy.shape[0] < 1000:
        raise RuntimeError("Too few points after slicing. Try adjusting --zmin/--zmax or check fused.ply quality.")

    # --- Rasterize to occupancy ---
    occ_raw, meta = rasterize_xy_fast(xy, res=args.res, margin=args.margin)
    cv2.imwrite(str(out_dir / "floorplan_raw.png"), occ_raw)

    # --- Morphology cleanup (fast) ---
    def k(sz):
        sz = max(1, int(sz))
        return np.ones((sz, sz), np.uint8)

    img = occ_raw.copy()

    if args.dilate_iter > 0:
        img = cv2.dilate(img, k(args.dilate_k), iterations=int(args.dilate_iter))

    if args.close_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k(args.close_k), iterations=int(args.close_iter))

    if args.open_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k(args.open_k), iterations=int(args.open_iter))

    img = remove_small_components(img, min_area=int(args.min_area))

    cv2.imwrite(str(out_dir / "floorplan_clean.png"), img)

    # --- Optional: line overlay for "floorplan-like" look ---
    if args.draw_lines:
        lines_img = hough_lines_overlay(
            img,
            min_line_length=int(args.min_line_len),
            max_line_gap=int(args.max_line_gap),
            manhattan=(not args.no_manhattan),
        )
        cv2.imwrite(str(out_dir / "floorplan_lines.png"), lines_img)

    print("Saved:")
    print(" -", out_dir / "floorplan_raw.png")
    print(" -", out_dir / "floorplan_clean.png")
    if args.draw_lines:
        print(" -", out_dir / "floorplan_lines.png")
    print("Note: white=occupied(wall candidates), black=free")


if __name__ == "__main__":
    main()
