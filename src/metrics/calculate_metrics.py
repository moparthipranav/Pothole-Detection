import torch
import torch.nn as nn
from src.metrics.iou import calculate_iou

def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    """Calculate precision, recall, F1-score"""
    tp, fp, fn = 0, 0, 0

    for pred, gt in zip(predictions, ground_truth):
        # Handle different prediction formats
        if pred is None or gt is None:
            continue
            
        # Convert pred to list if it's a tensor
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy().tolist() if pred.dim() > 1 else []
        
        # Convert gt to list if it's a tensor
        if isinstance(gt, torch.Tensor):
            gt = gt.cpu().numpy().tolist() if gt.dim() > 0 else []
        elif not isinstance(gt, (list, tuple)):
            gt = []
        
        if len(gt) == 0:
            fp += len(pred) if isinstance(pred, (list, tuple)) else 0
            continue
        
        # Handle single prediction (not wrapped in list)
        if isinstance(pred, (list, tuple)) and len(pred) > 0:
            if not isinstance(pred[0], (list, tuple)):
                pred = [pred]
        else:
            pred = []
        
        matched_gt = set()
        for p_item in pred:
            # Handle different box formats
            if isinstance(p_item, (list, tuple)) and len(p_item) >= 4:
                p_box = p_item[:4]
                p_conf = p_item[4] if len(p_item) > 4 else 0.5
            else:
                continue
            
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt_box in enumerate(gt):
                if gt_idx in matched_gt:
                    continue
                
                # Handle different gt box formats
                if isinstance(gt_box, (list, tuple)) and len(gt_box) >= 4:
                    gt_box = gt_box[:4]
                else:
                    continue
                
                iou = calculate_iou(p_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1
        
        fn += len(gt) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}



