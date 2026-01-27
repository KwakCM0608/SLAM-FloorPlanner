# SLAM_FloorPlan

휴대폰으로 촬영한 실내 영상을 입력으로 받아
COLMAP 기반 3D 재구성을 수행하고, 이를 바탕으로 2D 실내 평면도를 생성하는 프로젝트이다.

본 프로젝트는 연구 및 프로토타입 목적이며
현재는 2D 평면도 생성에 초점을 맞추고 있다.

---

## 프로젝트 목적

단일 스마트폰 영상만으로
별도의 깊이 센서 없이
사람이 이해할 수 있는 실내 평면도를 생성할 수 있는지 검증한다.

이를 위해 Structure-from-Motion(SfM)과
Multi-View Stereo(MVS) 기반 파이프라인을 구성했다.

---

## 파이프라인 개요

입력 영상부터 평면도 생성까지의 처리 흐름은 다음과 같다.

Input video →
Frame extraction →
COLMAP sparse reconstruction (SfM) →
COLMAP dense reconstruction (MVS) →
3D point cloud (fused.ply) →
바닥 평면 추정 및 정렬 →
높이 슬라이스 (벽 영역) →
XY 평면 투영 →
후처리 →
2D 평면도 이미지

---

## 사용 환경

* OS: Ubuntu 22.04 LTS
* GPU: NVIDIA GPU (CUDA 사용)
* Language: Python

주요 라이브러리:

* COLMAP
* Open3D
* OpenCV
* NumPy

대용량 중간 산출물은 GitHub에 포함하지 않는다.

---

## 디렉토리 구조

```
slam_floorplan/
├── scripts/
│   ├── extract_frames.py
│   ├── run_pipeline.py
│   └── make_floorplan.py
│
├── work/        # COLMAP 작업 디렉토리 (Git 제외)
├── output/      # 결과 이미지 (Git 제외)
│
├── README.md
└── .gitignore
```

---

## 결과물

**floorplan_raw.png**

3D 포인트클라우드를 그대로 상부에서 투영한 결과이다.
노이즈, 가구, 카메라 이동 잔상이 포함되어 있으며
파이프라인 상태를 확인하기 위한 용도로 사용한다.

**floorplan_clean.png**

floorplan_raw에 후처리를 적용한 결과이다.
형태학 연산과 연결 성분 필터링을 통해
사람이 보기 쉬운 평면도 형태를 목표로 한다.

색상 의미:

* 흰색: 점이 존재하는 영역 (벽 또는 장애물 후보)
* 검은색: 빈 공간

---

## 실행 방법 (요약)

가상환경 활성화:

```
source .venv/bin/activate
```

평면도 생성:

```
python scripts/make_floorplan.py --ply work/dense/fused.ply --out output
```

---

## 현재 한계

* Dense reconstruction 품질에 크게 의존함
* 벽과 가구가 명확히 분리되지 않음
* 관측 경로가 강하게 남는 경우 평면도가 왜곡될 수 있음
* 실시간 처리에는 적합하지 않음
* Depth Camera를 사용하지 않아 전체적인 성능이 떨어짐

---

## 향후 개선 방향

* Sparse reconstruction 및 카메라 궤적 기반 평면도 생성
* Manhattan world 가정 적용
* 문 및 통로 자동 감지
* SVG / DXF 벡터 도면 출력
* 학습 기반 평면도 추출 기법 적용
* Depth Camera 사용

---

## 작성자

곽창민

---
