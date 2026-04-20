import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SingleObjectDetector(nn.Module):
    """
    ResNet18 Backbone 기반 Single Object Detector.
    Classification(객체 종류)과 Regression(바운딩 박스)을 동시 예측.

    노트북 구현 대비 개선 사항:
      - CNN을 처음부터 학습하는 노트북 모델(Conv2d × 3 + FC) 대비,
        ImageNet 사전학습 ResNet18을 Backbone으로 채택하여 특징 추출 성능 향상.
      - He Initialization + BatchNorm1d 적용으로 학습 초기 수렴 안정화.
      - Regression Head 마지막에 Sigmoid를 추가하여 bbox 출력을 0~1로 정규화.
        (노트북의 raw 픽셀 좌표 출력 방식과 달리 이미지 크기에 독립적인 예측 가능)
    """
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.3):
        super(SingleObjectDetector, self).__init__()
        
        # 1. Backbone (Pre-trained ResNet18, 마지막 FC 레이어 제외)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 2. BBox Regression Head
        # 출력: [cx, cy, w, h] — Sigmoid로 0~1 정규화된 좌표
        # IoU 계산 시 DetectionOps.xywh_to_xyxy()로 변환 후 사용할 것
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4),
            nn.Sigmoid()  # 좌표 정규화 (0~1), Loss 스케일: ~0~1 범위의 MSE
        )
        
        # 3. Classification Head
        # 출력: 클래스 수 만큼의 logit (CrossEntropyLoss 입력용, Softmax 미적용)
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Custom Head에 대한 가중치 초기화 (He Initialization)
        self._initialize_weights(self.regressor)
        self._initialize_weights(self.classifier)

    def _initialize_weights(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): 입력 이미지 텐서 [B, C, H, W]
        Returns:
            tuple:
                bbox_preds  (Tensor): [B, 4] — Sigmoid 정규화된 [cx, cy, w, h]
                class_preds (Tensor): [B, num_classes] — 분류 logits
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1)  # [B, 512]
        
        bbox_preds = self.regressor(features)    # [B, 4], range: 0~1
        class_preds = self.classifier(features)  # [B, num_classes]
        
        return bbox_preds, class_preds


class DetectionLoss(nn.Module):
    """
    Single Object Detection을 위한 Multi-task Loss 함수.

    노트북의 핵심 학습 구조인 MSE Loss + CrossEntropy Loss 결합을
    재사용 가능한 모듈로 분리.
    alpha, beta 가중치로 두 손실 항의 균형을 조절할 수 있음.

    Note:
        bbox_targets는 모델 출력(Sigmoid 정규화 좌표)과 동일한 스케일이어야 함.
        DataLoader에서 bbox 좌표를 이미지 크기로 나누어 0~1로 정규화한 뒤 전달.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super(DetectionLoss, self).__init__()
        self.alpha = alpha  # BBox Regression Loss 가중치
        self.beta = beta    # Classification Loss 가중치
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        bbox_preds: torch.Tensor,    # [B, 4] — 모델 예측 bbox (0~1 정규화)
        class_preds: torch.Tensor,   # [B, num_classes] — 모델 예측 logits
        bbox_targets: torch.Tensor,  # [B, 4] — 정답 bbox (0~1 정규화 필수)
        class_targets: torch.Tensor  # [B] — 정답 클래스 인덱스 (Long 타입)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple:
                total_loss (Tensor): alpha * loss_bbox + beta * loss_cls
                loss_bbox  (Tensor): MSE Loss (bbox 위치 오차)
                loss_cls   (Tensor): CrossEntropy Loss (분류 오차)
        """
        # 1. Bounding Box 위치 오차 (Regression)
        loss_bbox = self.mse_loss(bbox_preds, bbox_targets)
        
        # 2. 객체 분류 오차 (Classification)
        loss_cls = self.ce_loss(class_preds, class_targets)
        
        # 3. Total Loss (Multi-task 가중합)
        total_loss = (self.alpha * loss_bbox) + (self.beta * loss_cls)
        return total_loss, loss_bbox, loss_cls
