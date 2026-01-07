import torch.nn as nn
import torch
from src.model.C2fModule import Conv

class DetectionHead(nn.Module):
    """
    Anchor-free YOLO-style detection head.
    Outputs raw box + class predictions per scale.
    """
    def __init__(self, nc=80, ch=()):
        super().__init__()

        self.nc = nc
        self.no = nc + 4  # box(4) + cls
        self.nl = len(ch) # number of detection layers

        assert self.nl == 4, "This head expects 4 feature maps (P2–P5)"

        # strides must correspond to P2, P3, P4, P5
        self.stride = torch.tensor([4, 8, 16, 32], dtype=torch.float)

        # Classification branch
        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, nc, 1)
            )
            for c in ch
        )

        # Regression branch
        self.box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(c, c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, 4, 1)
            )
            for c in ch
        )

    def forward(self, feats):
        """
        feats: list of feature maps [P2, P3, P4, P5]
        """
        outputs = []

        for i in range(self.nl):
            cls_map = self.cls_convs[i](feats[i])  # (B, nc, H, W)
            box_map = self.box_convs[i](feats[i])  # (B, 4, H, W)

            outputs.append((box_map, cls_map))

        return outputs
