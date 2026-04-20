import torch
import numpy as np
from typing import Union, List

class DetectionOps:
    """객체 탐지 평가 및 후처리를 위한 코어 연산 모듈
    
    노트북 구현 대비 개선 사항:
      - cv2.dnn.NMSBoxes 의존성 제거 → 순수 NumPy 기반 Greedy NMS 직접 구현
      - Tensor / NumPy / List 입력을 모두 수용하는 범용 인터페이스 설계
      - ZeroDivisionError 방지를 위한 Epsilon(1e-16) 처리 추가
    """
    
    @staticmethod
    def xywh_to_xyxy(
        box: Union[List[float], np.ndarray, torch.Tensor],
        normalized: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        [x_center, y_center, width, height] 포맷을
        [x_min, y_min, x_max, y_max] 포맷으로 변환.

        모델의 Regression 출력(Sigmoid로 정규화된 cxywh)과
        IoU 연산의 입력 포맷(xyxy)을 맞추기 위한 필수 유틸리티.
        
        Args:
            box:        [cx, cy, w, h] 형식의 좌표 배열
            normalized: True이면 입력이 0~1 범위의 정규화 좌표임을 의미 (모델 출력 기본값).
                        False이면 픽셀 단위 절대 좌표로 처리.
        Returns:
            [x_min, y_min, x_max, y_max] 형식으로 변환된 좌표
        """
        # normalized 여부는 호출자가 인지하면 충분하며,
        # 변환 수식 자체는 동일하게 적용됩니다.
        if isinstance(box, torch.Tensor):
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            x2 = box[0] + box[2] / 2
            y2 = box[1] + box[3] / 2
            return torch.stack([x1, y1, x2, y2])
        else:  # Numpy or List
            box = np.asarray(box, dtype=float)
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            x2 = box[0] + box[2] / 2
            y2 = box[1] + box[3] / 2
            return np.array([x1, y1, x2, y2])

    @staticmethod
    def calculate_iou(
        box1: Union[List[float], np.ndarray, torch.Tensor],
        box2: Union[List[float], np.ndarray, torch.Tensor]
    ) -> float:
        """
        두 Bounding Box 간의 IoU(Intersection over Union)를 계산.
        
        노트북 구현(calculate_iou)과 동일한 수식을 사용하되,
        Tensor 입력을 GPU/CPU 독립적으로 처리하고
        ZeroDivisionError 방지를 위한 Epsilon을 추가한 강화 버전.

        Args:
            box1, box2: [x1, y1, x2, y2] 형식의 좌표 (정규화 or 픽셀 단위 모두 가능)
        Returns:
            float: 0.0 ~ 1.0 사이의 IoU 값
        """
        if isinstance(box1, torch.Tensor):
            box1 = box1.detach().cpu().numpy()
        if isinstance(box2, torch.Tensor):
            box2 = box2.detach().cpu().numpy()

        box1 = np.asarray(box1, dtype=float)
        box2 = np.asarray(box2, dtype=float)

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        # ZeroDivisionError 방지를 위한 Epsilon 추가 (노트북 구현 개선)
        return float(intersection / (union + 1e-16))

    @staticmethod
    def nms(
        boxes: Union[np.ndarray, torch.Tensor],
        scores: Union[np.ndarray, torch.Tensor],
        iou_threshold: float = 0.5
    ) -> List[int]:
        """
        Greedy Non-Maximum Suppression (NMS) 순수 구현체.

        노트북에서는 cv2.dnn.NMSBoxes를 사용하였으나,
        본 구현은 외부 라이브러리 의존 없이 NumPy만으로 동일 로직을 재현.
        Confidence Score 기준 내림차순 정렬 후,
        IoU가 iou_threshold 이상인 중복 박스를 순차 제거.

        Args:
            boxes:         [N, 4] 형식의 바운딩 박스 배열 (x1, y1, x2, y2)
            scores:        [N] 형식의 신뢰도 점수 배열
            iou_threshold: 이 값 이상으로 겹치는 박스를 중복으로 판단하여 제거 (기본값: 0.5)
        Returns:
            List[int]: 최종적으로 유지할 박스의 인덱스 목록
        """
        if len(boxes) == 0:
            return []

        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()

        boxes = np.asarray(boxes, dtype=float)
        # [N, 1] 형태의 2차원 배열이 입력될 경우를 대비해 1차원으로 평탄화
        scores = np.asarray(scores, dtype=float).flatten()

        idxs = np.argsort(scores)[::-1]
        keep = []

        while len(idxs) > 0:
            current = idxs[0]
            keep.append(int(current))

            if len(idxs) == 1:
                break

            # 현재 박스와 나머지 모든 박스의 IoU를 일괄 계산
            ious = np.array([
                DetectionOps.calculate_iou(boxes[current], boxes[j])
                for j in idxs[1:]
            ])

            # IoU가 임계값 미만인(충분히 겹치지 않는) 박스만 다음 루프로 전달
            idxs = idxs[1:][ious < iou_threshold]

        return keep
