import torch
import torch.nn as nn
import torch.nn.functional as F

def dist2bbox(dist, points):
    x1 = points[:, 0] - dist[:, 0]
    y1 = points[:, 1] - dist[:, 1]
    x2 = points[:, 0] + dist[:, 2]
    y2 = points[:, 1] + dist[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=-1)

def bbox_ciou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.T
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter + eps
    iou = inter / union

    cx1 = (b1_x1 + b1_x2) / 2
    cy1 = (b1_y1 + b1_y2) / 2
    cx2 = (b2_x1 + b2_x2) / 2
    cy2 = (b2_y1 + b2_y2) / 2

    c2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    enclose_w = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    enclose_h = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2_enclose = enclose_w ** 2 + enclose_h ** 2 + eps

    v = (4 / torch.pi ** 2) * (
        torch.atan((b2_x2 - b2_x1) / (b2_y2 - b2_y1 + eps)) -
        torch.atan((b1_x2 - b1_x1) / (b1_y2 - b1_y1 + eps))
    ) ** 2

    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (c2 / c2_enclose + alpha * v)

class DFL(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred, target):
        """
        pred:   [N, 4, reg_max+1]
        target: [N, 4]
        """
        target = target.clamp(0, self.reg_max - 1e-4)

        left = target.long()
        right = left + 1

        wl = right.float() - target
        wr = target - left.float()

        pred = pred.view(-1, self.reg_max + 1)

        loss = (
            F.cross_entropy(pred, left.view(-1), reduction="none").view_as(left) * wl +
            F.cross_entropy(
                pred,
                right.clamp(max=self.reg_max).view(-1),
                reduction="none"
            ).view_as(left) * wr
        )

        return loss.mean()


def task_aligned_assigner(pred_boxes, pred_cls, gt_boxes, gt_labels,
                          topk=10, alpha=0.5, beta=6.0):

    N, C = pred_cls.shape
    M = gt_boxes.shape[0]

    if M == 0:
        return (
            torch.zeros(N, dtype=torch.bool, device=pred_boxes.device),
            torch.zeros_like(pred_boxes),
            torch.zeros_like(pred_cls)
        )

    ious = torch.zeros(N, M, device=pred_boxes.device)
    for i in range(M):
        ious[:, i] = bbox_ciou(
            pred_boxes,
            gt_boxes[i].unsqueeze(0).expand_as(pred_boxes)
        )

    # Get class scores for each GT label
    # pred_cls.sigmoid() has shape [N, C], gt_labels has shape [M]
    # We need to get the score for each prediction-gt pair
    pred_cls_sigmoid = pred_cls.sigmoid()  # [N, C]
    cls_scores = torch.zeros(N, M, device=pred_boxes.device)
    for i in range(M):
        cls_scores[:, i] = pred_cls_sigmoid[:, gt_labels[i]]
    
    align = (cls_scores ** alpha) * (ious ** beta)

    fg_mask = torch.zeros(N, dtype=torch.bool, device=pred_boxes.device)
    target_boxes = torch.zeros_like(pred_boxes)
    target_scores = torch.zeros_like(pred_cls)

    for i in range(M):
        topk_idx = align[:, i].topk(min(topk, N)).indices
        fg_mask[topk_idx] = True
        target_boxes[topk_idx] = gt_boxes[i]
        target_scores[topk_idx, gt_labels[i]] = ious[topk_idx, i]

    return fg_mask, target_boxes, target_scores



class YOLOLoss(nn.Module):
    def __init__(self, num_classes, reg_max=16):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max

        self.cls_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.dfl = DFL(reg_max)

        # Tuned for P2 pothole detection
        self.lambda_box = 9.0
        self.lambda_dfl = 2.0
        self.lambda_cls = 0.3

    def forward(self, outputs, targets):
        """
        outputs: dict with keys p2, p3, p4, p5
        targets: list of dicts (len = batch)
        """
        total_loss = 0.0
        batch_size = len(targets)

        for b in range(batch_size):

            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]

            for head in outputs.values():

                pred_dist, pred_cls, points = head

                # select batch index
                pred_dist = pred_dist[b].view(-1, 4, self.reg_max + 1)
                pred_cls = pred_cls[b].view(-1, self.num_classes)

                # decode distances
                prob = F.softmax(pred_dist, dim=-1)
                dist = (prob * torch.arange(
                    self.reg_max + 1, device=pred_dist.device
                )).sum(dim=-1)

                pred_boxes = dist2bbox(dist, points)

                fg_mask, tgt_boxes, tgt_scores = task_aligned_assigner(
                    pred_boxes, pred_cls, gt_boxes, gt_labels
                )

                if fg_mask.sum() == 0:
                    continue

                # IoU loss
                iou_loss = (1 - bbox_ciou(
                    pred_boxes[fg_mask],
                    tgt_boxes[fg_mask]
                )).mean()

                # DFL target distances
                tgt_dist = torch.cat([
                    points[fg_mask] - tgt_boxes[fg_mask][:, :2],
                    tgt_boxes[fg_mask][:, 2:] - points[fg_mask]
                ], dim=1)

                dfl_loss = self.dfl(pred_dist[fg_mask], tgt_dist)

                cls_loss = self.cls_loss(pred_cls, tgt_scores)

                total_loss += (
                    self.lambda_box * iou_loss +
                    self.lambda_dfl * dfl_loss +
                    self.lambda_cls * cls_loss
                )

        return total_loss / batch_size


