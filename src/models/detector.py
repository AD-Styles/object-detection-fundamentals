import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class SingleObjectDetector(nn.Module):
    """
    ResNet18 Backbone 기반 Single Object Detector.
    Classification(객체 종류)과 Regression(바운딩 박스)을 동시 예측합니다.
    """
    def __init__(self, num_classes: int = 1, dropout_rate: float = 0.3):
        super(SingleObjectDetector, self).__init__()
        
        # 1. Backbone (Pre-trained)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # 2. BBox Regression Head (x, y, w, h)
        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 4),
            nn.Sigmoid() # 좌표 정규화 (0~1)
        )
        
        # 3. Classification Head
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

    def _initialize_weights(self, module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): Input image tensor [B, C, H, W]
        Returns:
            tuple: (bbox_preds, class_preds)
        """
        features = self.backbone(x)
        features = torch.flatten(features, 1) # [B, 512]
        
        bbox_preds = self.regressor(features)
        class_preds = self.classifier(features)
        
        return bbox_preds, class_preds

class DetectionLoss(nn.Module):
    """
    Single Object Detection을 위한 Multi-task Loss 함수
    원본 노트북의 핵심인 MSE Loss와 CrossEntropy Loss의 결합을 모듈화합니다.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        super(DetectionLoss, self).__init__()
        self.alpha = alpha # BBox Loss 가중치
        self.beta = beta   # Class Loss 가중치
        
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, bbox_preds, class_preds, bbox_targets, class_targets):
        # 1. Bounding Box 위치 오차 (Regression)
        loss_bbox = self.mse_loss(bbox_preds, bbox_targets)
        
        # 2. 객체 분류 오차 (Classification)
        loss_cls = self.ce_loss(class_preds, class_targets)
        
        # 3. Total Loss
        total_loss = (self.alpha * loss_bbox) + (self.beta * loss_cls)
        return total_loss, loss_bbox, loss_cls
