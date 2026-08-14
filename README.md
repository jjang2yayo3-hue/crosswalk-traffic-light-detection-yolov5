# 🚸 School Zone Safety Monitoring System

> **YOLOv5, ByteTrack, and Risk Analysis for School Zone Crosswalk Monitoring**

어린이 보호구역의 안전을 향상시키기 위해 개발한 컴퓨터 비전 기반 안전 모니터링 시스템입니다.

본 프로젝트는 일반 횡단보도뿐만 아니라 노란 횡단보도를 탐지할 수 있도록 모델을 추가 학습하였으며,
차량·보행자·신호등을 동시에 분석하여 위험도를 계산합니다.


# 📌 Overview

기존 공개 모델은 일반 횡단보도 탐지에는 우수한 성능을 보였지만, 어린이 보호구역의 노란 횡단보도는 충분히 인식하지 못했습니다.

이를 해결하기 위해

* 노란 횡단보도 데이터 직접 수집
* 데이터 라벨링
* 데이터 증강
* YOLOv5 Fine-tuning

을 수행하였으며,

최종적으로는 여러 객체 탐지 모델과 객체 추적 알고리즘을 통합하여 실시간 위험도 분석 시스템을 구현하였습니다.

---

# ✨ Features

### 🚶 Object Detection

* 일반 횡단보도 탐지
* 노란 횡단보도 탐지
* 차량 탐지
* 보행자 탐지
* 보행 신호등 탐지

### 🚗 Object Tracking

* ByteTrack 기반 객체 추적
* 객체 ID 유지
* 차량 이동 속도 추정
* 이동 차량 / 정지 차량 구분

### ⚠ Risk Analysis

* 보행 신호 상태 분석
* 횡단보도 위 차량 여부 판단
* 보행자 접근 여부 판단
* 실시간 Risk Score 계산
* SAFE / CAUTION / DANGER 상태 표시

### 📈 Detection Stabilization

순간적인 오검출을 줄이기 위해 최근 프레임의 탐지 결과를 누적하여 상태를 안정화합니다.

* Crosswalk History
* Traffic Signal History
* Vehicle History
* Person History

---

# 🏗 System Architecture

Input Video
      │
      ▼
Frame Capture
      │
 ┌───────────────┐
 │               │
 ▼               ▼
COCO          exp4
 │             │
 │             ├─ Crosswalk
 │             └─ Signal
 │
 ▼
Vehicle / Person Detection
 │
 ▼
ByteTrack
 │
 ▼
Vehicle Speed Estimation
 │
 └──────────────┐
                ▼
            exp124
        Yellow Crosswalk
                │
                ▼
      Crosswalk Selection
                │
                ▼
      Temporal Filtering
                │
      ┌─────────┼─────────┐
      ▼         ▼         ▼
 Signal     Vehicle      Person
                │
                ▼
          Risk Engine
                │
                ▼
     SAFE / CAUTION / DANGER
                │
                ▼
       Output Visualization
--- 
### Processing Pipeline

입력 영상에서 COCO YOLOv5를 이용해 차량과 보행자를 탐지하고,
ByteTrack을 통해 객체를 프레임 간 추적합니다. 차량의 위치 변화를 이용해
이동 및 정지 상태를 판단하며, exp4와 최종 Fine-tuning 모델인 exp124를
통해 횡단보도와 노란 횡단보도를 탐지합니다.

이후 여러 프레임의 탐지 결과를 종합하여 안정적인 상태를 판단하고,
신호등 상태, 차량 및 보행자 상태 등의 정보를 Risk Engine에 전달합니다.
최종적으로 위험도를 계산하여 **SAFE / CAUTION / DANGER**로 표시합니다.

# ⚙ Development Environment

| Category         | Description |
| ---------------- | ----------- |
| Language         | Python 3.11 |
| Framework        | PyTorch     |
| Detection Model  | YOLOv5      |
| Tracking         | ByteTrack   |
| Image Processing | OpenCV      |
| Dataset Labeling | Roboflow    |
| Operating System | Windows 11  |

---

# 📂 Project Structure

.
├── data/
├── models/
├── runs/
│   ├── exp4/
│   └── exp124/
├── integrated_pipeline.py
├── detect.py
├── train.py
├── cross.py
├── class.py
├── coco_bytetrack.py
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

## Original Dataset

* 일반 횡단보도
* 보행 신호등

## Additional Dataset

직접 수집한 어린이 보호구역 노란 횡단보도 이미지

## Annotation

Roboflow를 이용하여 Bounding Box 라벨링을 수행하였습니다.

## Data Augmentation

* Horizontal Flip
* Rotation
* Brightness Adjustment
* Exposure Adjustment

데이터 다양성을 확보하여 다양한 환경에서의 탐지 성능을 향상시켰습니다.

---

# 🤖 Model Configuration

본 시스템은 하나의 모델이 아닌 3개의 모델을 통합하여 동작합니다.

| Model       | Purpose             |
| ----------- | ------------------- |
| COCO YOLOv5 | 차량 및 보행자 탐지         |
| exp4        | 일반 횡단보도 및 보행 신호등 탐지 |
| exp124      | 노란 횡단보도 탐지          |

각 모델의 결과를 통합하여 최종 위험도를 계산합니다.

---

# 🚀 Risk Engine

위험도는 다음 정보를 종합하여 계산됩니다.

* 🔴 보행 신호가 적색인가?
* 🚗 차량이 횡단보도 위를 이동 중인가?
* 🚙 차량이 횡단보도 위에 정차 중인가?
* 🚶 보행자가 횡단보도에 접근하는가?
* 🚦 신호등이 정상적으로 검출되었는가?

계산 결과는 다음과 같이 출력됩니다.

| Status  | Description |
| ------- | ----------- |
| SAFE    | 안전          |
| CAUTION | 주의          |
| DANGER  | 위험          |

---

# 📈 Training Result

| Model  | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| ------ | --------: | -----: | ------: | -----------: |
| exp124 |     0.982 |  0.982 |   0.994 |        0.884 |

노란 횡단보도 데이터를 추가 학습함으로써 기존 모델 대비 탐지 성능을 향상시켰습니다.

---

# ▶ How to Run

## Clone Repository

```bash
git clone https://github.com/your-repository.git
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python integrated_pipeline.py
```

---

# 📷 Result

실행 결과에서는 다음 정보를 동시에 확인할 수 있습니다.

* 일반 횡단보도
* 노란 횡단보도
* 차량
* 보행자
* 보행 신호등
* 객체 ID
* 차량 속도
* Risk Score
* SAFE / CAUTION / DANGER 상태

> [demo] (https://youtube.com/shorts/-nadjR90L3s?si=ibd-i2y7VL6G5jk1)

---

# 🔮 Future Work

* YOLOv8 기반 성능 비교
* 다양한 기상환경 데이터 추가
* Homography 기반 거리 추정
* 신호등 인식 정확도 향상
* 실시간 경고 시스템 구축
* Edge Device 최적화

---

# 📚 Tech Stack

* Python
* PyTorch
* YOLOv5
* ByteTrack
* OpenCV
* NumPy
* Supervision
* Roboflow

---

# 👨‍💻 Author

Computer Vision Project

**Keywords**

`YOLOv5` · `Computer Vision` · `Object Detection` · `ByteTrack` · `Risk Analysis` · `School Zone` · `Crosswalk Detection` · `Traffic Light Detection`
