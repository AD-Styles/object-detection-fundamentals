# 📝 Object Detection from Scratch: Learning the Core Logic
### 객체 탐지(Object Detection) 의 핵심 원리

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Torchvision](https://img.shields.io/badge/Torchvision-Latest-orange.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)

---

## 📌 프로젝트 요약 (Project Overview)
이번 프로젝트에서는 "컴퓨터가 이미지 속 물체의 위치를 어떻게 찾아내는 걸까?"라는 원초적인 궁금증을 바탕으로 단순히 이미 만들어진 라이브러리를 가져다 쓰는 것에 그치지 않고, 그 내부에서 돌아가는 핵심 로직들을 하나씩 직접 구현하며 객체 탐지의 '기초 체력'을 기르는 것을 목표로 삼았습니다. 바운딩 박스를 계산하는 아주 기초적인 수학부터 시작해서, 여러 개의 박스 중 가장 정확한 것만 남기는 알고리즘, 그리고 실제로 물체를 찾아내는 신경망을 설계하는 과정까지를 차례대로 담고 있습니다.

---

### ✍️ 주요 학습 포인트
| 단계 | 주요 내용 | 한 마디로? |
| :--- | :--- | :--- |
| **1. 기초 연산** | 두 박스가 얼마나 겹치는지 계산하는 IoU(Intersection over Union) 로직 구현 | "얼마나 정확하게 겹쳤나?" |
| **2. 결과 정제** | 중복된 예측 박스들을 하나로 정리하는 NMS(Non-Maximum Suppression) 알고리즘 구현 | "지저분한 박스들 정리하기" |
| **3. 모델 설계** | 이미지 특징을 추출하고 위치(좌표)와 종류(클래스)를 동시에 맞추는 신경망 구성 | "진짜 탐지기 만들기" |

---

## 🎯 핵심 목표 (Motivation)
| 핵심 질문 (Question) | 해결하기 위한 노력 (Approach) | 깨달은 점 (Insight) |
| :--- | :--- | :--- |
| **"박스가 겹친 정도를 어떻게 수치화할까?"** | IoU(Intersection over Union) 계산식을 직접 함수로 구현 | 컴퓨터가 이미지 좌표를 인식하고 면적을 계산하는 기초 기하학 원리 이해 |
| **"지저분하게 겹친 박스들을 어떻게 치울까?"** | NMS(Non-Maximum Suppression) 알고리즘 로직 설계 | 수많은 예측 중 가장 높은 확률만 남기는 효율적인 후처리 데이터 필터링 방식 체득 |
| **"하나의 신경망이 두 가지 일을 할 수 있을까?"** | 위치(Regression)와 분류(Classification)를 동시에 수행하는 모델 구조 설계 | 서로 다른 성격의 정답(좌표와 이름)을 함께 학습시키는 멀티태스크 학습의 기초 확립 |

---

## 📂 프로젝트 구조 (Project Structure)
```text
├── src/
│   ├── engine/
│   │   └── ops.py          # IoU, NMS 등 핵심 수학 연산 모듈
│   └── models/
│       └── detector.py     # 탐지 모델 아키텍처 및 손실 함수 정의
├── .gitignore              # 불필요한 파일 업로드 방지
├── LICENSE                 # MIT License (AD-Styles)
├── README.md               # 프로젝트 리포트
└── requirements.txt        # 라이브러리 설치 목록
```

---

## 🛠️ 핵심 구현 기술 (Technical Implementation)
| 핵심 개념 (Core Concept) | 구현 세부 내용 (Implementation Details) | 학습 포인트 및 통찰 (Key Insight) |
| :--- | :--- | :--- |
| **IoU 계산 알고리즘** | 두 박스의 교집합 면적을 합집합 면적으로 나누는 `calculate_iou` 함수 직접 구현. 분모가 0이 되는 에러 방지를 위해 `1e-16`(Epsilon) 추가. | 컴퓨터가 픽셀 좌표계를 이용해 면적과 겹침 정도를 수치화하는 원리를 이해함. |
| **NMS 후처리 로직** | Confidence Score가 가장 높은 박스를 기준으로 일정 수치 이상 겹치는 중복 박스들을 제거하는 `nms` 함수 설계. | 모델이 예측한 수많은 데이터 중 가장 신뢰도 높은 결과만 남기는 효율적인 데이터 정제 프로세스 습득. |
| **Multi-task 신경망** | ResNet18을 특징 추출기(Backbone)로 사용하고, 위치 예측(Regression)과 종류 판별(Classification)을 위한 각각의 Head 설계. | 서로 성격이 다른 정답(좌표와 클래스)을 하나의 모델이 동시에 학습할 수 있도록 Loss 함수를 설계하는 감각을 익힘. |
| **학습 최적화 기술** | 레이어 직후 `BatchNorm1d` 배치 및 커스텀 레이어에 `He Initialization`(가중치 초기화) 적용. | 딥러닝 모델의 학습 수렴 속도를 높이고 성능을 안정화하는 실무 엔지니어링 기법 체득. |

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
