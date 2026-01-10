import torch
import torch.nn as nn

class DebugTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train_step(self, image, target):
        self.model.train()

        preds = self.model(image)

        # Use ONLY the finest scale for debugging
        box_pred, cls_pred = preds[0]   # [B,4,H,W], [B,C,H,W]

        # Pick ONE grid cell (center)
        _, _, H, W = box_pred.shape
        i, j = H // 2, W // 2

        # Predicted ltrb
        pred_ltrb = box_pred[0, :, i, j]

        # Ground truth → ltrb relative to that cell
        stride = 4  # must match scale 0
        xc = (j + 0.5) * stride
        yc = (i + 0.5) * stride

        x1, y1, x2, y2 = target[0, 1:]  # assuming ONE GT box

        gt_ltrb = torch.tensor([
            xc - x1,
            yc - y1,
            x2 - xc,
            y2 - yc
        ], device=image.device)

        loss = nn.functional.l1_loss(pred_ltrb, gt_ltrb)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


