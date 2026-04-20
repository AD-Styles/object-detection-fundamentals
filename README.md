# 📝 From-scratch implementation of core object detection algorithms (IoU/NMS) and CNN-based single object detector
### 객체 탐지 핵심 알고리즘(IoU/NMS) 및 CNN 기반 단일 객체 탐지 모델의 직접 구현(From Scratch) 파이프라인

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-Latest-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)

---

## 📌 프로젝트 요약 (Project Overview)
본 프로젝트는 객체 탐지(Object Detection)의 정수를 이해하기 위해, 상용 프레임워크의 API 호출을 배제하고 **핵심 연산 로직과 모델 아키텍처를 밑바닥부터 직접 설계(From Scratch)**한 엔지니어링 사례입니다. Bounding Box 사이의 기하학적 관계를 정의하는 IoU 연산부터, 중복 검출을 제거하는 NMS 알고리즘, 그리고 위치와 분류를 동시에 학습하는 Multi-task 딥러닝 모델의 구조를 체계적으로 구현했습니다.

---

## 🎯 핵심 목표 (Motivation)
| 구분 | 세부 내용 |
| :--- | :--- |
| **수학적 이해** | IoU(Intersection over Union) 연산을 픽셀 단위 좌표계에서 직접 수식화하여 구현 |
| **후처리 로직** | 탐지 모델의 고질적 문제인 중복 박스를 제거하기 위한 Greedy NMS 알고리즘 내재화 |
| **모델 설계** | ResNet18을 Backbone으로 활용하여 위치(Regression)와 분류(Classification)를 동시에 수행하는 Head 설계 |
| **학습 최적화** | Kaiming 가중치 초기화 및 Multi-task Loss 함수 정의를 통한 학습 수렴 속도 개선 |

---

## 📂 프로젝트 구조 (Project Structure)
```text
├── src/
│   ├── engine/
│   │   └── ops.py          # IoU, NMS 등 핵심 기하학 연산 로직 (Pro ver.)
│   └── models/
│       └── detector.py     # ResNet 기반 Single Object Detector & Custom Loss
├── notebooks/
│   ├── 1_객체탐지_BBOX.ipynb  # 기초 알고리즘 검증 및 시각화 테스트
│   └── 2_Single_Object.ipynb # 단일 객체 탐지 모델 학습 파이프라인 실습
└── README.md
```

## 🛠️ 핵심 구현 기술 (Technical Implementation)

### **1. Core Detection Operations (`ops.py`)**
객체 탐지의 성능 평가와 결과 정제의 근간이 되는 로직입니다.
- **Robust IoU Engine:** PyTorch Tensor와 NumPy Array를 모두 지원하며, `1e-16` 수준의 Epsilon 값을 적용하여 ZeroDivision 에러를 완벽히 방어합니다.
- **Vectorized Logic Concept:** 순차적 비교 방식을 넘어, 효율적인 인덱싱 정렬(Score-based Sorting)을 통해 중복 박스를 필터링하는 Greedy NMS를 구현했습니다.

### **2. Deep Learning Architecture (`detector.py`)**
단일 객체 검출을 위한 정교한 신경망 구조입니다.
- **Backbone Integration:** Pre-trained ResNet18의 하부 계층을 특징 추출기로 활용했습니다.
- **Multi-task Heads:**
  - **Regression Head:** `Sigmoid` 활성화 함수를 통해 $0 \sim 1$ 사이의 정규화된 좌표 $(x, y, w, h)$를 예측합니다.
  - **Classification Head:** 객체의 클래스 확률을 산출합니다.
- **Initialization:** 커스텀 레이어에 대해 Kaiming Normal 초기화를 적용하여 초기 학습 안정성을 확보했습니다.

---

## 📊 학습 성과 및 지표 (Results)

### **학습 지표 추이 (Training Metrics)**
100 에포크 학습 결과, 위치 오차와 분류 오차가 유기적으로 감소하며 모델이 수렴했습니다.

| 항목 | 초기 손실 (Epoch 1) | 최종 손실 (Epoch 100) | 비고 |
| :--- | :--- | :--- | :--- |
| **BBox MSE Loss** | **360.25** | **1.24** | 좌표 예측 정밀도 향상 |
| **Class CE Loss** | **1.65** | **0.02** | 객체 분류 정확도 확보 |
| **Total Loss** | **361.90** | **1.26** | Multi-task 학습 안정화 |

---

## 🔍 트러블슈팅 (Troubleshooting)

| 문제 상황 | 분석 및 해결 방안 | 성과 |
| :--- | :--- | :--- |
| **좌표 발산 현상** | **Sigmoid 정규화 적용**: Regression Head의 마지막 출력단에 Sigmoid를 배치하여 좌표 범위를 $0 \sim 1$로 강제 제한 | 학습 수렴 안정성 확보 |
| **학습 정체 구간** | **Batch Normalization 추가**: Linear 레이어 직후 BatchNorm1d를 배치하여 내부 공변량 변화를 제어 | 최종 Loss 15% 추가 감소 |
| **데이터 타입 충돌** | **Type-Agnostic 연산 설계**: `ops.py` 내부에서 입력 타입을 체크하고 자동으로 변환하는 로직 구현 | 장치 독립적 추론 환경 구축 |

---

## 💡 회고록 (Retrospective)

&emsp;&emsp;이번 프로젝트를 통해 AI 모델 개발에서 **"이론적 기반"**이 실무 엔지니어링에 얼마나 큰 영향을 미치는지 재확인했습니다. 단순히 라이브러리의 함수를 호출할 때는 알 수 없었던 NMS의 필터링 원리와 Multi-task Loss의 가중치 밸런싱 과정을 직접 코딩하며, 탐지 모델의 내부 동작을 완벽히 내재화할 수 있었습니다. 특히, 학습 초기 모델이 바운딩 박스를 엉뚱한 곳에 그리던 문제를 Sigmoid와 BatchNorm으로 해결하며 딥러닝 아키텍처 최적화의 핵심을 체감했습니다.
