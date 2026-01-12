import torch
import torch.nn as nn

from src.model.YoloModel import YOLOModel
from src.Loss.yolo_loss import YOLOLoss

class ModelTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLOModel(
            base_channels=config["base_channels"],
            base_depth=config["base_depth"],
            n_classes=config["n_classes"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"]
        )

        self.criterion = YOLOLoss(num_classes=config["n_classes"])

    def train_step(self, images, targets):
        images = images.to(self.device)

        # ---- FIX TARGET FORMAT HERE ----
        # targets expected as list of dicts with "boxes" (xyxy format) and "labels"
        # Input format: YOLO format (cx, cy, w, h) normalized coordinates

        fixed_targets = []
        img_h, img_w = images.shape[2], images.shape[3]

        def convert_yolo_to_xyxy(boxes_yolo, img_h, img_w):
            """Convert YOLO format (cx, cy, w, h) normalized to xyxy format"""
            if boxes_yolo.shape[0] == 0:
                return boxes_yolo
            
            # Denormalize
            cx = boxes_yolo[:, 0] * img_w
            cy = boxes_yolo[:, 1] * img_h
            w = boxes_yolo[:, 2] * img_w
            h = boxes_yolo[:, 3] * img_h
            
            # Convert to xyxy
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            return torch.stack([x1, y1, x2, y2], dim=1)

        if isinstance(targets, torch.Tensor):
            # Case: [B, N, 5] - from collate_fn with batch index prepended
            if targets.dim() == 2 and targets.shape[1] == 6:
                # Format: [batch_idx, class, cx, cy, w, h]
                for b in range(len(set(targets[:, 0].tolist())) if targets.shape[0] > 0 else 0):
                    mask = targets[:, 0] == b
                    t = targets[mask]
                    if t.shape[0] == 0:
                        fixed_targets.append({
                            "boxes": torch.zeros((0, 4), device=self.device),
                            "labels": torch.zeros((0,), dtype=torch.long, device=self.device)
                        })
                    else:
                        boxes_yolo = t[:, 2:6]  # (cx, cy, w, h)
                        labels = t[:, 1].long()
                        
                        # Clamp labels to valid range [0, num_classes-1]
                        labels = torch.clamp(labels, 0, self.model.n_classes - 1)
                        
                        boxes_xyxy = convert_yolo_to_xyxy(boxes_yolo, img_h, img_w)
                        fixed_targets.append({
                            "boxes": boxes_xyxy.to(self.device),
                            "labels": labels.to(self.device)
                        })
            
            # Case: [N, 5] - single batch
            elif targets.dim() == 2 and targets.shape[1] == 5:
                boxes_yolo = targets[:, 1:5]
                labels = targets[:, 0].long()
                labels = torch.clamp(labels, 0, self.model.n_classes - 1)
                boxes_xyxy = convert_yolo_to_xyxy(boxes_yolo, img_h, img_w)
                fixed_targets.append({
                    "boxes": boxes_xyxy.to(self.device),
                    "labels": labels.to(self.device)
                })
            
            else:
                raise ValueError("Unsupported targets tensor shape")

        # Case: list of tensors
        elif isinstance(targets, list):
            for t in targets:
                if t.shape[0] == 0:
                    fixed_targets.append({
                        "boxes": torch.zeros((0, 4), device=self.device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=self.device)
                    })
                else:
                    boxes_yolo = t[:, 1:5]
                    labels = t[:, 0].long()
                    labels = torch.clamp(labels, 0, self.model.n_classes - 1)
                    boxes_xyxy = convert_yolo_to_xyxy(boxes_yolo, img_h, img_w)
                    fixed_targets.append({
                        "boxes": boxes_xyxy.to(self.device),
                        "labels": labels.to(self.device)
                    })

        else:
            raise TypeError("Unsupported targets type")

        # --------------------------------

        outputs = self.model(images)

        loss = self.criterion(outputs, fixed_targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()
