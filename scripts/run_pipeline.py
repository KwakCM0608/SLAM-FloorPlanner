import shutil
import subprocess
import argparse
from pathlib import Path

def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)

def extract_frames(video_path, frames_dir, fps):
    frames_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_dir.glob("*.jpg"):
        f.unlink()
    run(["ffmpeg", "-y", "-i", str(video_path), "-vf", f"fps={fps}", str(frames_dir / "frame_%05d.jpg")])

def colmap_sparse_to_ply(frames_dir, work_dir, use_gpu=1):
    """
    Run COLMAP sparse reconstruction only (no dense).
    Output: work_dir/sparse_points.ply
    """
    db_path = work_dir / "colmap.db"
    sparse_dir = work_dir / "sparse"

    # clean
    if db_path.exists():
        db_path.unlink()
    shutil.rmtree(sparse_dir, ignore_errors=True)

    sparse_dir.mkdir(parents=True, exist_ok=True)

    # 1) Feature extraction
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--ImageReader.camera_model", "OPENCV",
        "--SiftExtraction.use_gpu", str(use_gpu),
    ])

    # 2) Matching
    run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db_path),
        "--SiftMatching.use_gpu", str(use_gpu),
    ])

    # 3) Mapping
    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse_dir),
    ])

    model0 = sparse_dir / "0"
    if not model0.exists():
        raise RuntimeError("Sparse reconstruction failed: sparse/0 not found (No good initial pair 가능).")

    # 4) Convert sparse model to PLY
    sparse_ply = work_dir / "sparse_points.ply"
    run([
        "colmap", "model_converter",
        "--input_path", str(model0),
        "--output_path", str(sparse_ply),
        "--output_type", "PLY",
    ])

    if not sparse_ply.exists():
        raise RuntimeError("model_converter failed: sparse_points.ply not found.")
    return sparse_ply

def try_pipeline(video_path, out_dir, work_dir, fps, use_gpu):
    frames_dir = work_dir / "frames"
    print(f"\n[STEP 1] Extract frames @ {fps} fps")
    extract_frames(video_path, frames_dir, fps)

    print("\n[STEP 2] COLMAP sparse reconstruction -> PLY")
    ply_path = colmap_sparse_to_ply(frames_dir, work_dir, use_gpu=use_gpu)
    print("PLY:", ply_path)

    print("\n[STEP 3] Make 2D floorplan")
    run([
        "python3", str(Path("scripts") / "make_floorplan.py"),
        "--ply", str(ply_path),
        "--out", str(out_dir),
        "--res", "0.04",
        "--voxel", "0.05",
        "--zmax", "1.8",
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="output")
    ap.add_argument("--work", default="work")
    ap.add_argument("--gpu", type=int, default=1, help="1: try GPU for SIFT/matching, 0: CPU only")
    ap.add_argument("--fps", type=float, default=2.0, help="Initial fps for frame extraction")
    ap.add_argument("--retry_fps", default="1,0.5", help="Comma-separated fps values to retry if mapper fails")
    args = ap.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    work_dir = Path(args.work)

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    fps_list = [args.fps] + [float(x.strip()) for x in args.retry_fps.split(",") if x.strip()]

    last_err = None
    for fps in fps_list:
        try:
            print("\n========================================")
            print(f"Running pipeline with fps={fps}")
            print("========================================")
            try_pipeline(video_path, out_dir, work_dir, fps=fps, use_gpu=args.gpu)
            print("\n✅ DONE")
            print(" -", out_dir / "floorplan_raw.png")
            print(" -", out_dir / "floorplan_clean.png")
            return
        except subprocess.CalledProcessError as e:
            last_err = e
            # mapper 실패는 여기로 올 확률이 높음
            print(f"\n⚠️ Pipeline failed at fps={fps}. Trying next fps if available...")
        except Exception as e:
            last_err = e
            print(f"\n⚠️ Pipeline error at fps={fps}: {e}. Trying next fps if available...")

    print("\n❌ All attempts failed.")
    raise SystemExit(last_err)

if __name__ == "__main__":
    main()

