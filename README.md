# 🏥 Health Eat — 경구약제 이미지 객체 검출

> 코드잇 스프린트 AI 엔지니어링 10기 초급 프로젝트  
> **평가 지표**: mAP@[0.75:0.95] | **최고 캐글 점수**: 0.97581

---

## 📌 프로젝트 개요

헬스케어 스타트업 **Health Eat**의 AI 엔지니어로서, 모바일 앱에서 사용자가 촬영한 알약 이미지를 인식하고 해당 약 정보를 제공하는 객체 검출 모델을 개발했습니다.

- 사진 속 최대 4개의 알약에 대해 **클래스(약품명)**와 **위치(바운딩 박스)**를 검출
- train 232장 / test 842장 / 56 클래스
- 이미지 해상도: 976 × 1280 px

---

## 🏆 최종 결과

| 항목 | 내용 |
|------|------|
| 최고 캐글 점수 | **0.97581** |
| 최종 모델 조합 | YOLOv8s + YOLOv8m + YOLOv11s + YOLOv8l |
| 앙상블 방식 | WBF (Weighted Boxes Fusion) |
| iou_thr / skip_box_thr | 0.55 / 0.20 |
| TTA | True |

---

## 📁 디렉토리 구조

```
medicine-object-detection-project/
├── docs/
│   └── Health_Eat_프로젝트_보고서_최종.pdf   # 최종 보고서
├── EDA/
│   ├── EDA_분석_보고서.md                    # EDA 분석 보고서
│   └── EDA_분석_보고서.pdf
├── models/                                   # .gitkeep (모델 가중치는 Google Drive 보관)
├── notebooks/
│   ├── day1_setup_eda.ipynb
│   ├── eda_preprocessing.ipynb
│   ├── day2_train_yolo_ensemble.ipynb
│   ├── day2_rtdetr_experiment.ipynb
│   ├── day2_faster_rcnn_experiment.ipynb
│   ├── day3_wbf_hyperparameter_tuning.ipynb
│   ├── day4_new_models_ensemble_experiments.ipynb
│   └── day5_softnms_new_models_final_ensemble.ipynb
├── src/                                      # .gitkeep
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🔬 실험 요약

### 모델별 성능

| 모델 | mAP50-95 | 최종 활용 |
|------|----------|----------|
| YOLOv8s | 0.985 | ✅ 최종 앙상블 |
| YOLOv8m | 0.990 | ✅ 최종 앙상블 |
| YOLOv8l | 0.991 | ✅ 최종 앙상블 |
| YOLOv11s | 0.989 | ✅ 최종 앙상블 |
| YOLOv11m | 0.948 | ❌ |
| YOLOv11l | 0.925 | ❌ 과적합 |
| RT-DETR-L | 0.020 | ❌ 소규모 데이터 부적합 |
| Faster R-CNN | Loss 0.085 | ❌ |

### 캐글 점수 흐름

| 날짜 | 실험 | 점수 |
|------|------|------|
| 4/29 | YOLOv8s 단독 첫 제출 | 0.95788 |
| 4/29 | WBF 앙상블 (4모델) | 0.97162 |
| 4/30 | 3모델 균등 가중치 WBF | 0.97292 |
| 5/1 | skip_box_thr=0.20 튜닝 | 0.97414 |
| 5/5 | YOLOv8l 추가 4모델 WBF | **0.97581** ✅ |

---

## ⚙️ 환경 설정

```bash
pip install ultralytics ensemble-boxes albumentations
```

```python
# 세션 시작 (Google Colab)
from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/pill_detection'
```

---

## 🚀 실행 방법

```python
# 1. YOLO 학습
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.train(data='yolo_data/data.yaml', epochs=100, imgsz=1024)

# 2. 예측 CSV 생성 → notebooks/03_train_yolo.ipynb 참고
# 3. WBF 앙상블   → notebooks/04_ensemble_submit.ipynb 참고
```

---

## 📄 제출물

- 📑 **[보고서 PDF 다운로드](./docs/Health_Eat_프로젝트_보고서_최종.pdf)**
- 📝 **[업무일지](https://www.notion.so/348404be4d178027becdf1f89cb87b16)**

---

## 💡 핵심 인사이트

- **imgsz=1024**: EDA에서 원본 해상도(976×1280) 확인 후 설정 → 각인 디테일 보존으로 성능 우위 확보
- **WBF > Soft-NMS**: Soft-NMS(0.96329) 실험 후 WBF 최종 채택
- **skip_box_thr=0.20**: 낮은 confidence 박스를 살리는 것이 작은 객체에 유리
- **모델 수보다 다양성**: 5~6모델 앙상블이 오히려 점수 하락 → 상호보완적 조합이 핵심
- **RT-DETR 한계**: Transformer 기반 모델은 소규모 데이터(232장)에서 학습 불가 수준

---

## 🛠️ 개발 환경

- Google Colab Pro (T4 GPU)
- Python 3.10 / PyTorch / Ultralytics YOLO
- Google Drive 기반 데이터 관리
