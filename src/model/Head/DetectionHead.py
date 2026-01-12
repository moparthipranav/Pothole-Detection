import torch
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, nc, ch, reg_max=16):
        super().__init__()

        self.nc = nc
        self.reg_max = reg_max
        self.no = 4 * (reg_max + 1) + nc

        self.heads = nn.ModuleList(
            nn.Conv2d(c, self.no, 1) for c in ch
        )
    def forward(self, features):
        outputs = []
        for feat, head in zip(features, self.heads):
            outputs.append(head(feat))  # <-- single tensor
        return tuple(outputs)

