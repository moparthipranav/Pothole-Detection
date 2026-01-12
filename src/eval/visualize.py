import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from utils.logger import logging

def visualize_predictions(images, predictions, ground_truth, output_dir="visualizations", max_images=5):
    """Visualize predicted and ground truth bounding boxes"""
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, (image, pred, gt) in enumerate(zip(images[:max_images], predictions[:max_images], ground_truth[:max_images])):
        # Skip if None
        if image is None or pred is None or gt is None:
            continue
        
        # Convert image to numpy
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Get image dimensions for scaling
        h, w, _ = img_np.shape

        # Draw ground truth boxes in green
        if isinstance(gt, (list, tuple)):
            for box in gt:
                # Assuming GT format is [batch_idx, class_id, x_center, y_center, width, height] (YOLO format)
                # Need to convert to [x1, y1, x2, y2]
                if isinstance(box, (list, tuple)) and len(box) >= 6:
                    # Denormalize coordinates
                    x_center, y_center, box_w, box_h = box[2] * w, box[3] * h, box[4] * w, box[5] * h
                    x1, y1 = int(x_center - box_w / 2), int(y_center - box_h / 2)
                    x2, y2 = int(x_center + box_w / 2), int(y_center + box_h / 2)
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_np, "GT", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw prediction boxes in blue with confidence scores
        if isinstance(pred, (list, tuple)):
            for p_item in pred:
                # Assuming prediction format is [x1, y1, x2, y2, confidence]
                if isinstance(p_item, (list, tuple)) and len(p_item) >= 5:
                    x1, y1, x2, y2, conf = int(p_item[0]), int(p_item[1]), int(p_item[2]), int(p_item[3]), p_item[4]
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img_np, f"Pred: {conf:.2f}", (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Save visualization
        output_path = os.path.join(output_dir, f"prediction_{idx}.jpg")
        cv2.imwrite(output_path, img_np)

def plot_training_metrics(losses, accuracies, output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

    # PLot loss
    ax1.plot(losses, label="Training Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if accuracies:
        epochs = range(1, len(accuracies) + 1)
        precisions = [m["precision"] for m in accuracies]
        recalls = [m["recall"] for m in accuracies]
        f1_scores = [m["f1"] for m in accuracies]
        
        ax2.plot(epochs, precisions, label="Precision", marker='o', linewidth=2)
        ax2.plot(epochs, recalls, label="Recall", marker='s', linewidth=2)
        ax2.plot(epochs, f1_scores, label="F1-Score", marker='^', linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_title("Evaluation Metrics Over Epochs")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(output_path, dpi=150)
    logging.info(f"Saved metrics plot to {output_path}")
    plt.close()
