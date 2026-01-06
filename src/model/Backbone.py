import torch
import torch.nn as nn
from src.model.C2fModule import Conv, C2f


class YOLOBackbone(nn.Module):
    def __init__(self, base_channels, base_depth):
        super().__init__()
        c = base_channels

        # Stem: 640 → 320
        self.stem = Conv(3, c, k=3, s=2)

        # Stage 1: 320 → 160
        self.stage1_conv = Conv(c, c * 2, k=3, s=2)
        self.stage1_c2f  = C2f(c * 2, c * 2, n=base_depth)

        # Stage 2: 160 → 80
        self.stage2_conv = Conv(c * 2, c * 4, k=3, s=2)
        self.stage2_c2f  = C2f(c * 4, c * 4, n=base_depth * 2)

        # Stage 3: 80 → 40
        self.stage3_conv = Conv(c * 4, c * 8, k=3, s=2)
        self.stage3_c2f  = C2f(c * 8, c * 8, n=base_depth * 2)

        # Stage 4: 40 → 20
        self.stage4_conv = Conv(c * 8, c * 16, k=3, s=2)
        self.stage4_c2f  = C2f(c * 16, c * 16, n=base_depth)

        # SPPF (keeps 20×20)
        self.sppf = SPPF(c * 16, c * 16, k=5)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1_c2f(self.stage1_conv(x))
        p2 = x                         # 160×160

        x = self.stage2_c2f(self.stage2_conv(x))
        p3 = x                         # 80×80

        x = self.stage3_c2f(self.stage3_conv(x))
        p4 = x                         # 40×40

        x = self.stage4_c2f(self.stage4_conv(x))
        p5 = self.sppf(x)              # 20×20

        print("p2:", p2.shape)
        print("p3:", p3.shape)
        print("p4:", p4.shape)
        print("p5:", p5.shape)

        return p2, p3, p4, p5
    
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv8."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
    