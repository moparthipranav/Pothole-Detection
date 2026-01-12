import torch
import torch.nn as nn

from src.model.backbone.Backbone import YOLOBackbone
from src.model.Neck.YoloNeck import YoloNeckHead
from src.model.Layers.DetectDecoder import DetectDecoder


class YOLOModel(nn.Module):
    """
    Full YOLO-style detector:
    Backbone → Neck+Head → (Decoder during inference)
    """


    def __init__(self, base_channels=32, base_depth=1, n_classes=1, reg_max=16):
        super().__init__()

        self.reg_max = reg_max
        self.n_classes = n_classes
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

    def _make_grid(self, h, w, stride, device):
        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )
        points = torch.stack([x, y], dim=-1).float()
        points = (points + 0.5) * stride
        return points.view(-1, 2)


    def forward(self, x, targets=None):
        device = x.device

        # 1. Backbone
        p2, p3, p4, p5 = self.backbone(x)

        # 2. Neck + Head
        preds = self.neck_head(p2, p3, p4, p5)
        # preds MUST be a tuple/list of 4 tensors (P2–P5)

        if self.training:
            outputs = {}

            strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
            names = ["p2", "p3", "p4", "p5"]
            feats = [p2, p3, p4, p5]

            for name, feat, pred in zip(names, feats, preds):
                # pred shape: [B, C, H, W]
                B, C, H, W = pred.shape

                # Split dist + cls
                reg_max = self.reg_max  # or store explicitly
                dist_ch = 4 * (reg_max + 1)

                dist, cls = torch.split(
                    pred,
                    [dist_ch, self.decoder.nc],
                    dim=1
                )

                # [B,C,H,W] → [B,H,W,C]
                dist = dist.permute(0, 2, 3, 1).contiguous()
                cls = cls.permute(0, 2, 3, 1).contiguous()

                points = self._make_grid(
                    H, W, strides[name], device
                )

                outputs[name] = (dist, cls, points)

            return outputs

        else:
            # Inference path: split preds into (box, cls) pairs
            split_preds = []
            reg_max = self.reg_max
            dist_ch = 4 * (reg_max + 1)
            
            for pred in preds:
                dist, cls = torch.split(
                    pred,
                    [dist_ch, self.decoder.nc],
                    dim=1
                )
                split_preds.append((dist, cls))
            
            boxes, scores = self.decoder(split_preds)
            return boxes, scores

        
        
