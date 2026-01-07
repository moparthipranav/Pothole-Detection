import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.C2fModule import C2f, Conv
from src.model.C2fAcmix import C2f_ACmix
from src.model.DetectionHead import DetectionHead


class YoloNeckHead(nn.Module):
    '''
    The Neck is the main component of the architecture which aggregates the features learned by the backbone and detection.
    The Neck is where the magic of multi-scale detection happens. This model uses PaNET(Path Aggregation Network) structure where the job 
    is to take high level semantic features from the deeper layers and inject them back into the high resolution features of the earlier layers.
    This ensures that even if an object is tiny, the model has enough context to detect it.

    There are two main flows:
    1. Upsampling
    2. Downsampling
    '''
    def __init__(self, base_channels, base_depth, n_classes=80):
        super().__init__()
        c = base_channels

        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Top-down
        self.c2f_p4_up = C2f(c * (16 + 8), c * 8, n=base_depth)
        self.c2f_p3_up = C2f(c * (8 + 4),  c * 4, n=base_depth)
        self.c2f_p2_up = C2f(c * (4 + 2),  c * 2, n=base_depth)

        # Bottom-up
        self.down_p2 = Conv(c * 2, c * 2, k=3, s=2)
        self.down_p3 = Conv(c * 4, c * 4, k=3, s=2)
        self.down_p4 = Conv(c * 8, c * 8, k=3, s=2)

        self.c2f_p3_down = C2f(c * (2 + 4),  c * 4, n=base_depth)
        self.c2f_p4_down = C2f_ACmix(c * (4 + 8),  c * 8, n=base_depth)
        self.c2f_p5_down = C2f_ACmix(c * (8 + 16), c * 16, n=base_depth)

        # ONE detection head
        self.detect_head = DetectionHead(
            nc=n_classes,
            ch=[c * 2, c * 4, c * 8, c * 16]
        )

    def forward(self, p2, p3, p4, p5):
        # Top-down
        p4_up = self.c2f_p4_up(torch.cat([self.up(p5), p4], dim=1))
        p3_up = self.c2f_p3_up(torch.cat([self.up(p4_up), p3], dim=1))
        p2_out = self.c2f_p2_up(torch.cat([self.up(p3_up), p2], dim=1))

        # Bottom-up
        p3_down = self.c2f_p3_down(
            torch.cat([self.down_p2(p2_out), p3_up], dim=1)
        )
        p4_down = self.c2f_p4_down(
            torch.cat([self.down_p3(p3_down), p4_up], dim=1)
        )
        p5_down = self.c2f_p5_down(
            torch.cat([self.down_p4(p4_down), p5], dim=1)
        )

        # Detection (multi-scale)
        return self.detect_head([p2_out, p3_down, p4_down, p5_down])
