import os
import torch
from src.data.data_ingestion import RoboFlowDataset
from src.train.trainer import ModelTrainer
from src.metrics.iou import calculate_iou
from src.eval.visualize import visualize_predictions, plot_training_metrics
from src.metrics.calculate_metrics import calculate_metrics
from torch.utils.data import DataLoader
from utils.logger import logging
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import numpy as np

def get_train_dataloader(config):
    """Load training dataset and return DataLoader"""
    train_dataset = RoboFlowDataset(
        img_dir=config["train_img_dir"],
        label_dir=config["train_label_dir"],
        img_size=config.get("img_size")
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size"),
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        collate_fn=collate_fn_yolo
    )
    
    return train_loader

def collate_fn_yolo(batch):
    """Custom collate function for YOLO dataset"""
    images = []
    targets = []
    
    for i, (img, boxes) in enumerate(batch):
        images.append(img)

        if boxes.numel() > 0:
            b = torch.zeros((boxes.shape[0], 6))
            b[:, 0] = i # The batch number
            b[:, 1:] = boxes
            targets.append(b)
        

    images = torch.stack(images, dim=0) # A new dimension occurs when using stack
    targets = torch.cat(targets, dim=0) if len(targets) else torch.zeros((0,6))

    return images, targets


def train_pipeline(config):
    """Main training pipeline"""
    logging.info("Starting training pipeline")
    
    # Load data
    train_loader = get_train_dataloader(config)
    logging.info(f"Loaded {len(train_loader)} batches")
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    logging.info(f"Model initialized on device: {trainer.device}")
    
    # Tracking metrics
    losses = []
    accuracies = []

    # Training loop
    num_epochs = config.get("epochs")
    batches_per_epoch = config.get("batches_per_epoch", None)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        all_predictions = []
        all_ground_truth = []
        batch_count = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            # Limit batches per epoch if specified
            if batches_per_epoch and batch_idx >= batches_per_epoch:
                break
            
            loss = trainer.train_step(images, targets)
            epoch_loss += loss
            batch_count += 1

            with torch.no_grad():
                outputs = trainer.model(images)
                predictions = outputs if isinstance(outputs, list) else [outputs]
                all_predictions.extend(predictions)
                all_ground_truth.extend(targets)
            
            if batch_idx % config.get("log_interval", 10) == 0:
                logging.info(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}] Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        losses.append(avg_loss)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")
        
        # Calculate metrics
        metrics = calculate_metrics(all_predictions, all_ground_truth)
        accuracies.append(metrics)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Metrics - Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1']:.4f}")
        
        # Save checkpoint
        if config.get("save_checkpoint", True):
            checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(trainer.model.state_dict(), checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Final visualizations
    logging.info("Generating visualizations...")
    plot_training_metrics(losses, accuracies)

    # Visualize final predictions on sample batch
    sample_images, sample_targets = next(iter(train_loader))

    # Ensure model is in eval mode so forward returns decoded boxes/scores
    trainer.model.eval()
    with torch.no_grad():
        # move images to device for inference
        imgs_device = sample_images.to(trainer.device)
        boxes, scores = trainer.model(imgs_device)  # boxes: [B, N, 4], scores: [B, N, nc]

        # Convert decoder outputs to list of [x1,y1,x2,y2,conf] per image for visualization
        preds_list = []
        # scores: [B, N, nc] -> max score per box
        max_scores, _ = scores.max(dim=-1)  # [B, N]
        for b in range(boxes.shape[0]):
            b_boxes = boxes[b].cpu().numpy()
            b_confs = max_scores[b].cpu().numpy()
            items = []
            for box, conf in zip(b_boxes, b_confs):
                x1, y1, x2, y2 = box.tolist()
                items.append([float(x1), float(y1), float(x2), float(y2), float(conf)])
            preds_list.append(items)

    # Normalize ground-truth format for visualization (list per image)
    gt_list = []
    for t in sample_targets:
        if isinstance(t, torch.Tensor):
            gt_list.append(t.cpu().numpy().tolist() if t.numel() else [])
        else:
            gt_list.append(t)

    visualize_predictions(sample_images, preds_list, gt_list)

    logging.info("Training pipeline completed")
    return trainer.model

if __name__ == "__main__":
    config = {
        "train_img_dir": "C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/images",
        "train_label_dir": "C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/labels",
        "img_size": 416,
        "batch_size": 8,
        "num_workers": 0,
        "epochs": 10, 
        "batches_per_epoch": 8,
        "log_interval": 2,
        "save_checkpoint": True,
        "checkpoint_dir": "checkpoints",
        "base_channels": 32,
        "base_depth": 1,
        "n_classes": 1,
        "lr": 0.001
    }
    
    train_pipeline(config)