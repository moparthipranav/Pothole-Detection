import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.C2fModule import Conv, C2f

class YOLOBackbone(nn.Module):
    def __init__(self, base_channels, base_depth):
        super().__init__()

        # Fitting the input 640 x 640 to 320 x 320
        self.stem = Conv(3, base_channels, k=3, s=2)

        # Stage 1: 320 x 320 -> 160 x 160 and Applying C2f
        self.stage1_conv = Conv(base_channels, 2 * base_channels, k=3, s=2)
        self.stage1_c2f = C2f(base_channels * 2, base_channels * 2, n=base_depth)

        # Stage 2 (160x160 -> 80x80)
        self.stage2_conv = Conv(base_channels * 2, base_channels * 4, k=3, s=2)
        self.stage2_c2f = C2f(base_channels * 4, base_channels * 4, n=base_depth * 2)
        
        # Stage 3 (80x80 -> 40x40)
        self.stage3_conv = Conv(base_channels * 4, base_channels * 8, k=3, s=2)
        self.stage3_c2f = C2f(base_channels * 8, base_channels * 8, n=base_depth * 2)
        
        # Stage 4 (40x40 -> 20x20)
        self.stage4_conv = Conv(base_channels * 8, base_channels * 16, k=3, s=2)
        self.stage4_c2f = C2f(base_channels * 16, base_channels * 16, n=base_depth)
        
        # Final SPPF Layer
        self.sppf = SPPF(base_channels * 16, base_channels * 16, k=5)

    def forward(self, x):
        x = self.stem(x)
        p1 = self.stage1_c2f(self.stage1_conv(x))
        p2 = self.stage2_c2f(self.stage2_conv(p1))
        p3 = self.stage3_c2f(self.stage3_conv(p2))
        p4 = self.stage4_c2f(self.stage4_conv(p3))
        p5 = self.sppf(self.stage4_c2f(p4))
        
        return p3, p4, p5  # We return multiple scales for the Neck/Head
    
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
    