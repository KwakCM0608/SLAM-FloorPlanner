#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import cv2
import struct


# ---------------- COLMAP binary readers (minimal) ----------------
def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            image_id = read_next_bytes(fid, 4, "I")[0]
            qw, qx, qy, qz = read_next_bytes(fid, 8*4, "dddd")
            tx, ty, tz = read_next_bytes(fid, 8*3, "ddd")
            camera_id = read_next_bytes(fid, 4, "I")[0]
            name_bytes = []
            while True:
                c = fid.read(1)
                if c == b"\x00" or c == b"":
                    break
                name_bytes.append(c)
            name = b"".join(name_bytes).decode("utf-8", errors="ignore")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * 24, 1)
            images[image_id] = {"q": np.array([qw,qx,qy,qz], np.float64),
                                "t": np.array([tx,ty,tz], np.float64),
                                "camera_id": camera_id,
                                "name": name}
    return images

def read_points3d_binary(path):
    points = []
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            _pid = read_next_bytes(fid, 8, "Q")[0]
            x, y, z = read_next_bytes(fid, 24, "ddd")
            _r, _g, _b = read_next_bytes(fid, 3, "BBB")
            _error = read_next_bytes(fid, 8, "d")[0]
            track_len = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(track_len * (4 + 4), 1)  # (image_id, point2D_idx)
            points.append((x, y, z))
    return np.array(points, dtype=np.float64)

def qvec2rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)

def camera_center_world(qvec, tvec):
    R = qvec2rotmat(qvec)
    return -R.T @ tvec

def pca_align_3d(P):
    # align principal components so "up" becomes smallest-variance axis
    mu = P.mean(axis=0, keepdims=True)
    X = P - mu
    cov = (X.T @ X) / max(len(X)-1, 1)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1]  # largest -> smallest
    V = V[:, idx]
    Xr = X @ V
    # make Z be smallest variance axis => use columns [0,1,2] already, but Z = Xr[:,2] is smallest variance
    return Xr, V, mu.squeeze()

def rasterize_xy_fast(xy, res, margin):
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    xmin -= margin; ymin -= margin
    xmax += margin; ymax += margin
    W = int(np.ceil((xmax-xmin)/res)); H = int(np.ceil((ymax-ymin)/res))
    W = max(W, 300); H = max(H, 300)
    ix = ((xy[:,0]-xmin)/res).astype(np.int32)
    iy = ((xy[:,1]-ymin)/res).astype(np.int32)
    valid = (ix>=0)&(ix<W)&(iy>=0)&(iy<H)
    ix = ix[valid]; iy = iy[valid]
    lin = iy.astype(np.int64)*np.int64(W)+ix.astype(np.int64)
    counts = np.bincount(lin, minlength=H*W).reshape(H,W)
    img = (counts>0).astype(np.uint8)*255
    img = np.flipud(img)
    return img

def remove_small_components(binary_255, min_area):
    if min_area <= 0:
        return binary_255
    bw = (binary_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(binary_255)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == i] = 255
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="COLMAP sparse model dir (e.g., work/sparse/0)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.8)

    # height slice in PCA-aligned coordinates (relative to z0)
    ap.add_argument("--zmin", type=float, default=0.2)
    ap.add_argument("--zmax", type=float, default=1.6)

    # morphology
    ap.add_argument("--dilate_k", type=int, default=3)
    ap.add_argument("--dilate_iter", type=int, default=1)
    ap.add_argument("--close_k", type=int, default=11)
    ap.add_argument("--close_iter", type=int, default=2)
    ap.add_argument("--open_k", type=int, default=7)
    ap.add_argument("--open_iter", type=int, default=1)
    ap.add_argument("--min_area", type=int, default=600)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Path(args.model)
    pts_path = model / "points3D.bin"
    img_path = model / "images.bin"
    if not pts_path.exists():
        raise FileNotFoundError(f"points3D.bin not found: {pts_path}")
    if not img_path.exists():
        raise FileNotFoundError(f"images.bin not found: {img_path}")

    P = read_points3d_binary(str(pts_path))
    if len(P) < 2000:
        raise RuntimeError(f"Too few sparse points: {len(P)} (reconstruction 품질이 낮을 수 있음)")
    print("Sparse points:", len(P))

    # Use camera centers to stabilize PCA (optional but helpful for scale/orientation sanity)
    imgs = read_images_binary(str(img_path))
    centers = []
    for v in imgs.values():
        centers.append(camera_center_world(v["q"], v["t"]))
    C = np.stack(centers, axis=0)

    # PCA on combined set (points + camera centers) improves axis robustness
    Pc = np.vstack([P, C])
    Pc_aligned, V, mu = pca_align_3d(Pc)

    # Split back
    P_aligned = Pc_aligned[:len(P), :]

    # Floor height estimate in aligned coords: low percentile of Z
    z = P_aligned[:,2]
    z0 = float(np.percentile(z, 5))
    keep = (z > z0 + args.zmin) & (z < z0 + args.zmax)
    xy = P_aligned[keep, :2]
    print("Kept points:", len(xy), "z0:", z0)

    if len(xy) < 1000:
        raise RuntimeError("Too few points after z-slice. Try widening z-range: e.g., --zmin 0.1 --zmax 2.0")

    raw = rasterize_xy_fast(xy, args.res, args.margin)
    cv2.imwrite(str(out_dir / "floorplan_raw.png"), raw)

    def k(sz):
        sz = max(1, int(sz))
        return np.ones((sz, sz), np.uint8)

    img = raw.copy()
    if args.dilate_iter > 0:
        img = cv2.dilate(img, k(args.dilate_k), iterations=int(args.dilate_iter))
    if args.close_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, k(args.close_k), iterations=int(args.close_iter))
    if args.open_iter > 0:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k(args.open_k), iterations=int(args.open_iter))
    img = remove_small_components(img, args.min_area)

    cv2.imwrite(str(out_dir / "floorplan_clean.png"), img)
    print("Saved:", out_dir / "floorplan_raw.png")
    print("Saved:", out_dir / "floorplan_clean.png")
    print("white=structure candidates, black=free")

if __name__ == "__main__":
    main()
