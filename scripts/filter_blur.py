from pathlib import Path
import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--in_dir", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--thr", type=float, default=80.0, help="lower -> keep more, higher -> stricter")
args = ap.parse_args()

in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

imgs = sorted(in_dir.glob("*.jpg"))
kept = 0

for p in imgs:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    score = cv2.Laplacian(img, cv2.CV_64F).var()  # focus measure
    if score >= args.thr:
        kept += 1
        cv2.imwrite(str(out_dir / p.name), cv2.imread(str(p)))
print(f"Total: {len(imgs)}, Kept: {kept}, Threshold: {args.thr}")

