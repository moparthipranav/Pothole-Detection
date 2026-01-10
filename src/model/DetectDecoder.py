import torch
import torch.nn as nn

class DetectDecoder(nn.Module):
    '''
    DetectDecoder is responsible for the bounding box predictions and calculating the class probabilities
    '''

    def __init__(self, nc=80, strides=(4,8,16,32)):
        super().__init__()
        self.nc = nc
        self.strides = strides

    def forward(self, preds):
        """
        preds: list of tuples [(box, cls), ...] for each scale
               box: [B, 4, H, W]
               cls: [B, nc, H, W]
        """

        all_boxes = []
        all_scores = []

        for (box, cls), stride in zip(preds, self.strides):
            B,_,H,W = box.shape
            device = box.device

            box = box.relu()
            cls = cls.sigmoid()

            y, x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )

            xc = (x + 0.5) * stride
            yc = (y + 0.5) * stride

            x1 = xc - box[:, 0]
            y1 = yc - box[:, 1]
            x2 = xc + box[:, 2]
            y2 = yc + box[:, 3]

            boxes = torch.stack([x1, y1, x2, y2], dim = 1)

            boxes = boxes.view(B, 4, -1).permute(0, 2, 1)   # [B, HW, 4]
            scores = cls.view(B, self.nc, -1).permute(0, 2, 1)  # [B, HW, nc]

            all_boxes.append(boxes)
            all_scores.append(scores)

        all_boxes = torch.cat(all_boxes, dim=1)
        all_scores = torch.cat(all_scores, dim=1)

        return all_boxes, all_scores
    