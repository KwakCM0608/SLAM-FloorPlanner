#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
import open3d as o3d

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def rotation_matrix_from_vectors(a, b):
    # Rotate vector a to vector b
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = np.linalg.norm(v)
    if s < 1e-12:
        # Parallel or anti-parallel
        if c > 0:
            return np.eye(3, dtype=np.float64)
        # 180deg: pick any orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v = v / (np.linalg.norm(v) + 1e-12)
        # Rodrigues for 180deg: R = I + 2K^2 (since sin=0, 1-cos=2)
        K = np.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + 2 * (K @ K)

    # Rodrigues
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]], dtype=np.float64)
    R = np.eye(3) + K + (K @ K) * ((1 - c) / (s * s))
    return R

def count_grid_xy(xy, res, margin=0.5):
    mn = xy.min(axis=0) - margin
    mx = xy.max(axis=0) + margin
    size = np.ceil((mx - mn) / res).astype(int) + 1
    W, H = int(size[0]), int(size[1])

    ij = np.floor((xy - mn) / res).astype(int)
    # clamp
    ij[:, 0] = np.clip(ij[:, 0], 0, W - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, H - 1)

    counts = np.zeros((H, W), dtype=np.int32)
    # vectorized bincount
    idx = ij[:, 1] * W + ij[:, 0]
    bc = np.bincount(idx, minlength=H * W)
    counts = bc.reshape(H, W).astype(np.int32)
    return counts, mn, (W, H)

def normalize_uint8(img_f):
    vmin = float(np.min(img_f))
    vmax = float(np.max(img_f))
    if vmax - vmin < 1e-9:
        return np.zeros_like(img_f, dtype=np.uint8)
    out = (255.0 * (img_f - vmin) / (vmax - vmin)).astype(np.uint8)
    return out

def fill_holes(bin_img):
    # bin_img: 0/255
    h, w = bin_img.shape[:2]
    inv = (255 - bin_img).copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(inv, mask, (0, 0), 0)  # fill background in inverted
    holes = inv  # remaining white are holes in original
    filled = cv2.bitwise_or(bin_img, holes)
    return filled

def keep_components(bin_img, min_area, keep_largest_fallback=True):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bin_img > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bin_img

    areas = stats[1:, cv2.CC_STAT_AREA]
    ids = np.arange(1, num)

    keep = ids[areas >= min_area]
    if len(keep) == 0 and keep_largest_fallback:
        # keep the largest component even if min_area too large
        keep = [ids[int(np.argmax(areas))]]

    out = np.zeros_like(bin_img)
    for i in keep:
        out[labels == i] = 255
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Input fused pointcloud PLY (e.g., fused.ply)")
    ap.add_argument("--out", default="~/slam_floorplan/output", help="Output folder (default: ~/slam_floorplan/output)")
    ap.add_argument("--res", type=float, default=0.05, help="Grid resolution in meters/pixel (default 0.05)")
    ap.add_argument("--margin", type=float, default=0.5, help="Extra margin around bbox in meters")
    ap.add_argument("--voxel", type=float, default=0.03, help="Voxel downsample size in meters (default 0.03)")
    ap.add_argument("--ransac_dist", type=float, default=0.02, help="RANSAC distance threshold for floor plane (m)")
    ap.add_argument("--floor_keep_ratio", type=float, default=0.25, help="Use this ratio of lowest points for floor search (0~1)")
    ap.add_argument("--zmin", type=float, default=0.10, help="Min height above floor to keep (m)")
    ap.add_argument("--zmax", type=float, default=1.80, help="Max height above floor to keep (m)")

    ap.add_argument("--blur", type=int, default=5, help="Gaussian blur kernel (odd). 0 disables.")
    ap.add_argument("--thr_mode", choices=["auto", "otsu", "percentile", "fixed"], default="auto")
    ap.add_argument("--thr_percentile", type=float, default=80.0, help="Percentile for threshold on counts (nonzero)")
    ap.add_argument("--thr_fixed", type=int, default=3, help="Fixed threshold on raw counts if thr_mode=fixed")

    ap.add_argument("--close_k", type=int, default=9)
    ap.add_argument("--close_iter", type=int, default=2)
    ap.add_argument("--open_k", type=int, default=5)
    ap.add_argument("--open_iter", type=int, default=1)
    ap.add_argument("--min_area", type=int, default=800, help="Min component area (pixels) to keep")

    args = ap.parse_args()

    ply = os.path.expanduser(args.model)
    out_dir = os.path.expanduser(args.out)
    ensure_dir(out_dir)

    print(f"[1] Loading: {ply}")
    if not os.path.exists(ply):
        raise FileNotFoundError(ply)

    pcd = o3d.io.read_point_cloud(ply)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        raise RuntimeError("Empty point cloud")

    # Downsample for stable plane fit (and speed)
    if args.voxel > 0:
        pcd_ds = pcd.voxel_down_sample(voxel_size=float(args.voxel))
    else:
        pcd_ds = pcd
    pts_ds = np.asarray(pcd_ds.points)
    if pts_ds.size == 0:
        raise RuntimeError("Empty point cloud after downsample")

    # --- pick low slice for floor search (reduces chance of picking ceiling) ---
    z = pts_ds[:, 2]
    q = np.quantile(z, np.clip(args.floor_keep_ratio, 0.05, 1.0))
    low_mask = z <= q
    low_pts = pts_ds[low_mask]
    if low_pts.shape[0] < 500:
        low_pts = pts_ds  # fallback

    low_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low_pts))

    print("[2] Plane segmentation (floor) ...")
    plane_model, inliers = low_pcd.segment_plane(distance_threshold=float(args.ransac_dist),
                                                 ransac_n=3,
                                                 num_iterations=2000)
    a, b, c, d = plane_model
    n = np.array([a, b, c], dtype=np.float64)
    n_norm = np.linalg.norm(n) + 1e-12
    n = n / n_norm

    # Ensure normal points "up" (positive z in current frame)
    if n[2] < 0:
        n = -n
        d = -d

    # Rotate so floor normal aligns with +Z
    R = rotation_matrix_from_vectors(n, np.array([0.0, 0.0, 1.0], dtype=np.float64))

    pts_rot = (R @ pts.T).T

    # After rotation, plane eq becomes z + d' = 0 approximately if normal -> +Z
    # Estimate floor z as median of rotated inlier points
    inlier_pts = np.asarray(low_pcd.select_by_index(inliers).points)
    inlier_rot = (R @ inlier_pts.T).T
    floor_z = float(np.median(inlier_rot[:, 2]))

    # Translate so floor is at z=0
    pts_aligned = pts_rot.copy()
    pts_aligned[:, 2] -= floor_z

    print(f"[3] Estimated floor z ≈ {floor_z:.3f} m (aligned frame)")
    # Save aligned ply for debugging
    aligned_ply = os.path.join(out_dir, "fused_aligned.ply")
    pcd_aligned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_aligned))
    o3d.io.write_point_cloud(aligned_ply, pcd_aligned, write_ascii=False, compressed=False)
    print(f"[4] Wrote aligned PLY: {aligned_ply}")

    # --- height filter (walls region) ---
    m = (pts_aligned[:, 2] >= float(args.zmin)) & (pts_aligned[:, 2] <= float(args.zmax))
    pts_h = pts_aligned[m]
    if pts_h.shape[0] < 2000:
        print("[WARN] Too few points after z-filter. Relax zmin/zmax or check floor fit.")
        # fallback: use a wider band
        m2 = (pts_aligned[:, 2] >= 0.0) & (pts_aligned[:, 2] <= max(2.5, float(args.zmax)))
        pts_h = pts_aligned[m2]

    # --- density grid ---
    xy = pts_h[:, :2]
    counts, mn, (W, H) = count_grid_xy(xy, res=float(args.res), margin=float(args.margin))

    # Density visualization (log)
    dens = np.log1p(counts.astype(np.float32))
    dens_u8 = normalize_uint8(dens)
    cv2.imwrite(os.path.join(out_dir, "floorplan_density.png"), dens_u8)

    # Optional blur to connect gaps
    dens_for_thr = dens_u8.copy()
    if args.blur and args.blur >= 3 and args.blur % 2 == 1:
        dens_for_thr = cv2.GaussianBlur(dens_for_thr, (args.blur, args.blur), 0)

    # --- threshold selection ---
    nz = counts[counts > 0]
    if nz.size == 0:
        raise RuntimeError("No occupied cells in grid. Something is wrong with filtering/alignment.")

    def thr_by_otsu(img_u8):
        # only on nonzero region to avoid black dominance
        mask = (img_u8 > 0).astype(np.uint8) * 255
        vals = img_u8[mask > 0]
        if vals.size < 200:
            return None
        # Otsu on values only
        # cv2.threshold expects an image; make a 1D image
        tmp = vals.reshape(-1, 1)
        t, _ = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return int(t)

    thr_u8 = None
    mode = args.thr_mode
    if mode in ["auto", "otsu"]:
        t = thr_by_otsu(dens_for_thr)
        if t is not None:
            thr_u8 = t
            mode = "otsu"
        elif mode == "otsu":
            mode = "percentile"  # fallback

    if thr_u8 is None and mode in ["auto", "percentile"]:
        # percentile on raw counts (more stable)
        p = float(np.clip(args.thr_percentile, 50.0, 99.9))
        thr_count = int(np.quantile(nz, p / 100.0))
        thr_count = max(thr_count, 1)
        # convert count-threshold to u8-space using dens_u8 mapping
        # approximate by applying same log then normalize using current vmin/vmax
        thr_log = np.log1p(thr_count)
        vmin = float(np.min(dens))
        vmax = float(np.max(dens))
        thr_u8 = int(np.clip(255.0 * (thr_log - vmin) / (vmax - vmin + 1e-9), 1, 254))

    if thr_u8 is None:
        # fixed in count-space
        thr_count = max(int(args.thr_fixed), 1)
        thr_log = np.log1p(thr_count)
        vmin = float(np.min(dens))
        vmax = float(np.max(dens))
        thr_u8 = int(np.clip(255.0 * (thr_log - vmin) / (vmax - vmin + 1e-9), 1, 254))

    thr_img = (dens_for_thr >= thr_u8).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(out_dir, "floorplan_thr.png"), thr_img)

    # --- morphology ---
    def k(n): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))
    img = thr_img

    if args.close_k > 0 and args.close_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k(int(args.close_k)), iterations=int(args.close_iter))
    if args.open_k > 0 and args.open_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k(int(args.open_k)), iterations=int(args.open_iter))

    img = fill_holes(img)
    cv2.imwrite(os.path.join(out_dir, "floorplan_morph.png"), img)

    # --- component filtering (safe fallback) ---
    clean = keep_components(img, min_area=int(args.min_area), keep_largest_fallback=True)
    cv2.imwrite(os.path.join(out_dir, "floorplan_clean.png"), clean)

    print("[DONE] Saved:")
    for fn in ["fused_aligned.ply", "floorplan_density.png", "floorplan_thr.png", "floorplan_morph.png", "floorplan_clean.png"]:
        print("  -", os.path.join(out_dir, fn))

if __name__ == "__main__":
    main()

