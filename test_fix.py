#!/usr/bin/env python3
"""Quick test to verify the training fix works"""
import os
import sys
import torch
sys.path.insert(0, os.path.dirname(__file__))

from src.data.data_ingestion import RoboFlowDataset
from src.train.trainer import ModelTrainer
from torch.utils.data import DataLoader

def collate_fn_yolo(batch):
    """Custom collate function for YOLO dataset"""
    images = []
    targets = []
    
    for i, (img, boxes) in enumerate(batch):
        images.append(img)
        if boxes.numel() > 0:
            b = torch.zeros((boxes.shape[0], 6))
            b[:, 0] = i  # The batch number
            b[:, 1:] = boxes
            targets.append(b)
    
    images = torch.stack(images, dim=0)
    targets = torch.cat(targets, dim=0) if len(targets) else torch.zeros((0, 6))
    return images, targets

# Test with 1 batch
config = {
    "train_img_dir": "C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/images",
    "train_label_dir": "C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/labels",
    "img_size": 416,
    "batch_size": 2,
    "num_workers": 0,
    "epochs": 1,
    "batches_per_epoch": 1,
    "log_interval": 1,
    "save_checkpoint": False,
    "base_channels": 32,
    "base_depth": 1,
    "n_classes": 1,
    "lr": 0.001
}

print("Loading dataset...")
dataset = RoboFlowDataset(
    img_dir=config["train_img_dir"],
    label_dir=config["train_label_dir"],
    img_size=config["img_size"]
)

loader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn_yolo
)

print("Initializing trainer...")
trainer = ModelTrainer(config)

print("Testing training step with 1 batch...")
for images, targets in loader:
    print(f"Image shape: {images.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets:\n{targets}")
    
    try:
        loss = trainer.train_step(images, targets)
        print(f"✓ Training step successful! Loss: {loss:.4f}")
        print("✓ Fix verified - no IndexError!")
    except Exception as e:
        print(f"✗ Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    break
