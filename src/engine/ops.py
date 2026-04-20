import torch
import numpy as np
from typing import Union, List

class DetectionOps:
    """객체 탐지 평가 및 후처리를 위한 코어 연산 모듈"""
    
    @staticmethod
    def xywh_to_xyxy(box: Union[List[float], np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        [x_center, y_center, width, height] 포맷을 
        [x_min, y_min, x_max, y_max] 포맷으로 변환합니다.
        모델의 Regression 출력과 IoU 연산의 호환성을 맞추기 위한 필수 유틸리티입니다.
        """
        if isinstance(box, torch.Tensor):
            x1 = box[0] - box[2] / 2
            y1 = box[1] - box[3] / 2
            x2 = box[0] + box[2] / 2
            y2 = box[1] + box[3] / 2
            return torch.stack([x1, y1, x2, y2])
        else: # Numpy or List
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
        두 Bounding Box 간의 IoU를 계산합니다. (Tensor 및 Numpy 모두 지원)
        Args:
            box1, box2: [x1, y1, x2, y2]
        Returns:
            float: 0.0 ~ 1.0 사이의 IoU 값
        """
        # Tensor 입력 시 GPU/CPU 독립적 처리를 위해 float 변환
        if isinstance(box1, torch.Tensor): box1 = box1.cpu().numpy()
        if isinstance(box2, torch.Tensor): box2 = box2.cpu().numpy()

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        # ZeroDivisionError 방지를 위한 epsilon 추가
        return float(intersection / (union + 1e-16))

    @staticmethod
    def nms(
        boxes: Union[np.ndarray, torch.Tensor], 
        scores: Union[np.ndarray, torch.Tensor], 
        iou_threshold: float = 0.5
    ) -> List[int]:
        """
        Greedy Non-Maximum Suppression (NMS) 구현체
        """
        if len(boxes) == 0:
            return []
            
        if isinstance(boxes, torch.Tensor): boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor): scores = scores.cpu().numpy()
            
        idxs = np.argsort(scores)[::-1]
        keep = []

        while len(idxs) > 0:
            current = idxs[0]
            keep.append(int(current))
            
            if len(idxs) == 1:
                break
                
            # Vectorized 형태의 개념을 차용하여 나머지 박스와의 IoU 일괄 계산
            ious = np.array([
                DetectionOps.calculate_iou(boxes[current], boxes[j]) 
                for j in idxs[1:]
            ])
            
            # IoU가 임계값 미만인(겹치지 않는) 인덱스만 추출하여 다음 루프 진행
            idxs = idxs[1:][ious < iou_threshold]
            
        return keep
