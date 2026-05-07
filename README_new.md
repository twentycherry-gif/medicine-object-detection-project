# 💊 경구약제 이미지 객체 검출 프로젝트

> **Health Eat** — 알약 인식 AI 모델 개발 및 성능 최적화  
> 코드잇 스프린트 AI 엔지니어링 10기 초급 프로젝트

---

## 🏆 최종 결과

| 항목 | 내용 |
|------|------|
| **Kaggle Public Score** | **0.97581** |
| 평가 지표 | mAP@[0.75:0.95] |
| 최적 모델 | YOLOv8s + YOLOv8m + YOLOv11s + YOLOv8l (WBF 앙상블) |
| 총 실험 모델 수 | 12개 |
| 총 캐글 제출 횟수 | 20회+ |

---

## 📋 프로젝트 개요

모바일 앱으로 알약 사진을 찍으면 약 이름과 위치를 자동으로 인식하는 객체 검출 모델을 개발합니다.

- **데이터**: AI Hub 경구약제 이미지 — train 232장 / test 842장 / 56개 클래스
- **목표**: 사진 속 최대 4개 알약의 클래스(이름)와 바운딩 박스를 동시에 검출
- **평가**: Kaggle Private Competition (mAP@[0.75:0.95])

---

## 📁 폴더 구조

```
medicine-object-detection-project/
├── docs/
│   ├── Health_Eat_프로젝트_보고서.pdf   ← 최종 보고서
│   └── 업무일지_최혜리.md               ← 업무일지
├── EDA/
│   └── eda_combined_최종.ipynb          ← EDA 통합 노트북
├── notebooks/
│   └── day1_setup_eda.ipynb             ← Day1 셋업 노트북
├── models/                              ← 모델 가중치 (용량 문제로 미포함)
├── src/                                 ← 유틸리티 코드
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📄 결과물

| 파일 | 링크 |
|------|------|
| 📑 보고서 PDF | [docs/Health_Eat_프로젝트_보고서.pdf](docs/Health_Eat_프로젝트_보고서.pdf) |
| 📝 업무일지 | [docs/업무일지_최혜리.md](docs/업무일지_최혜리.md) |

---

## 🔬 실험 요약

### 모델 실험 결과

| 모델 | mAP50-95 | 비고 |
|------|----------|------|
| YOLOv8s | 0.985 | 기준 모델 |
| YOLOv8m | 0.990 | |
| YOLOv11s | 0.989 | |
| YOLOv8l | 0.991 | ✅ 앙상블 핵심 |
| YOLOv11s (저lr) | 0.992 | |
| RT-DETR-L | 0.020 | 소규모 데이터 부적합 |

### 앙상블 점수 변화

| 단계 | Kaggle 점수 |
|------|------------|
| YOLOv8s 단독 | 0.95788 |
| 3모델 WBF 앙상블 | 0.97292 |
| 파라미터 튜닝 (iou=0.55) | 0.97414 |
| **4모델 WBF (v8l 추가)** | **0.97581 ✅** |

---

## ⚙️ 환경 설정

```bash
pip install -r requirements.txt
```

**주요 라이브러리**
```
ultralytics>=8.4.0
torch>=2.0.0
ensemble-boxes
albumentations
opencv-python
pandas
numpy<2
```

---

## 🚀 실행 방법

### 1. 환경 셋업 (Google Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/pill_detection'
!pip install ultralytics ensemble-boxes -q
```

### 2. 모델 학습

```python
from ultralytics import YOLO

model = YOLO('yolov8l.pt')
model.train(
    data=f'{BASE}/yolo_data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=8,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    device=0
)
```

### 3. WBF 앙상블 예측

```python
from ensemble_boxes import weighted_boxes_fusion

boxes, scores, labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=[1.0, 1.0, 1.0, 1.0],
    iou_thr=0.55,
    skip_box_thr=0.20
)
```

---

## 📊 핵심 인사이트

- **imgsz=1024** 설정이 초기 점수 차별화 핵심 요인
- **WBF > Soft-NMS**: 박스를 합치는 방식이 이 데이터셋에 적합
- **모델 다양성 > 모델 수**: 6모델(0.97410) < 4모델(0.97581)
- **과증강 역효과**: augv2의 train mAP는 높지만 캐글 점수 낮음
- **구조적 다양성**: YOLOv8l(43M) 추가로 0.97414 벽 돌파

---

## ⚠️ 한계점

- 소규모 데이터(232장)로 인한 과적합 위험
- 배경/조명 100% 단일 환경 → 실제 서비스 OOD 취약성
- K-Fold Cross Validation 미적용
- 실제 스마트폰 촬영 환경 Blind Test 미진행

---

## 👤 제출자

| 항목 | 내용 |
|------|------|
| 이름 | 최혜리 |
| 과정 | 코드잇 스프린트 AI 엔지니어링 10기 |
| 제출일 | 2026년 5월 10일 |
