import torch
import torch.nn as nn

from src.model.Backbone import YOLOBackbone
from src.model.YoloNeck import YoloNeckHead
from src.model.DetectDecoder import DetectDecoder


class YOLOModel(nn.Module):
    """
    Full YOLO-style detector:
    Backbone → Neck+Head → (Decoder during inference)
    """

    def __init__(self, base_channels=64, base_depth=2, n_classes=80):
        super().__init__()

        self.backbone = YOLOBackbone(
            base_channels=base_channels,
            base_depth=base_depth
        )

        self.neck_head = YoloNeckHead(
            base_channels=base_channels,
            base_depth=base_depth,
            n_classes=n_classes
        )

        self.decoder = DetectDecoder(
            nc=n_classes,
            strides=(4, 8, 16, 32)   # must match your backbone
        )

    def forward(self, x, targets=None):
        # 1. Backbone
        p2, p3, p4, p5 = self.backbone(x)

        # 2. Neck + Head (raw predictions)
        preds = self.neck_head(p2, p3, p4, p5)

        # 3. Training vs Inference
        if self.training:
            # Loss will consume raw predictions
            return preds
        else:
            # Decode boxes for inference
            boxes, scores = self.decoder(preds)
            return boxes, scores
        
        
