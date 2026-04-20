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
