# Indoor Floorplan Generation from Video (COLMAP)

스마트폰으로 촬영한 실내 동영상을 입력으로 받아
COLMAP 기반 3D 재구성(SfM / MVS)을 수행하고,
최종적으로 2D 실내 평면도(floorplan)를 생성하는 프로젝트입니다.

일반 RGB 영상만 사용하며, LiDAR나 Depth 카메라는 필요하지 않습니다.

---

## Pipeline

본 프로젝트는 다음 단계를 따릅니다.

1. 동영상에서 프레임 추출
2. COLMAP sparse reconstruction (카메라 포즈 추정)
3. COLMAP dense reconstruction (point cloud 생성)
4. Dense point cloud를 2D로 투영하여 평면도 생성

---

## Project Structure

```
slam-floorplan/
├─ scripts/
│  ├─ run_colmap_sparse.sh
│  ├─ run_colmap_dense.sh
│  └─ make_floorplan.py
├─ output/
│  ├─ floorplan_raw.png
│  ├─ floorplan_thr.png
│  └─ floorplan_clean.png
├─ work/
│  └─ dense/
│     └─ fused.ply
├─ README.md
└─ .gitignore
```

---

## Requirements

* Ubuntu 22.04
* NVIDIA GPU (CUDA 지원)
* COLMAP (CUDA enabled)
* Python 3.10
* ffmpeg

필요한 Python 라이브러리는 다음과 같습니다.

```
pip install open3d opencv-python numpy
```

---

## Usage

### 1. Frame extraction

입력 영상에서 프레임을 추출합니다.

```
ffmpeg -i input.mp4 -vf fps=1.2 work/frames/frame_%05d.jpg
```

> 입력 영상(mp4)은 GitHub에 포함되어 있지 않으며,
> 사용자가 직접 촬영한 영상을 사용해야 합니다.

---

### 2. Sparse reconstruction

카메라 포즈와 sparse point cloud를 생성합니다.

```
bash scripts/run_colmap_sparse.sh
```

---

### 3. Dense reconstruction

Dense point cloud를 생성합니다.

```
bash scripts/run_colmap_dense.sh
```

결과물은 다음 경로에 생성됩니다.

```
work/dense/fused.ply
```

---

### 4. Floorplan generation

Dense point cloud를 2D 평면도로 변환합니다.

```
python scripts/make_floorplan.py \
  --model work/dense/fused.ply \
  --out output
```

---

## Output

<img width="495" height="502" alt="Image" src="https://github.com/user-attachments/assets/c6bba252-eea1-430a-a67b-178e8b1a0671" />
<img width="1631" height="857" alt="Image" src="https://github.com/user-attachments/assets/3254070d-5d0c-4c1c-9602-76933532130c" />
<img width="891" height="706" alt="image" src="https://github.com/user-attachments/assets/68e7156d-9770-4dee-8392-07661f6b94d1" />

* `fused.ply`
  → 3D 포인트를 그대로 투영한 결과
  
* `floorplan_raw.png`
  → 3D 포인트를 그대로 투영한 결과 (2D형태로 출력)

* `floorplan_thr.png`
  → 밀도 기반 threshold 적용 결과

* `floorplan_clean.png`
  → morphology 연산으로 정제된 최종 평면도

---

## Notes

* 충분한 카메라 이동(parallax)이 필요합니다.
* 회전 위주 촬영은 구조 복원이 어렵습니다.
* 텍스처가 거의 없는 벽/바닥 영역은 인식이 약할 수 있습니다.
* Dense reconstruction 품질이 평면도 품질을 크게 좌우합니다.

---

## License

본 프로젝트는 연구 및 교육 목적을 위해 작성되었습니다.

---
