import torch
import torch.nn as nn

from src.model.Backbone import YOLOBackbone
from src.model.YoloNeck import YoloNeckHead

class YOLOModel(nn.Module):
    """
    Full YOLO-style model:
    Backbone → Neck + Head
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
    
    def forward(self, x):
        # Backbone outputs
        p2, p3, p4, p5 = self.backbone(x)

        # Neck + Head
        outputs = self.neck_head(p2, p3, p4, p5)

        return outputs