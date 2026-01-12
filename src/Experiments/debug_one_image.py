import torch
import os
from src.model.YoloModel import YOLOModel
from src.data.data_ingestion import RoboFlowDataset
from src.components.DebugTrainer import DebugTrainer
from utils.logger import logging

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load ONE image
    # Try to find dataset path
    dataset_paths = [
        "C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/images",
        "data/train/images",
        "../../../Downloads/Pothole Detection.v1i.yolov8/train/images"
    ]
    
    img_dir = None
    label_dir = None
    
    for path in dataset_paths:
        if os.path.exists(path):
            img_dir = path
            label_dir = path.replace("images", "labels")
            break
    
    if img_dir is None:
        raise FileNotFoundError("Could not find dataset. Please check dataset paths or set DATA_PATH environment variable")
    
    logging.info(f"Loading dataset from: {img_dir}")
    
    dataset = RoboFlowDataset(
        img_dir=img_dir,
        label_dir=label_dir
    )

    image, target = dataset[0]
    image = image.unsqueeze(0).to(device)
    target = target.to(device)
    logging.info(f"Loaded image shape: {image.shape}, target shape: {target.shape}")
    
    # 2. Model
    model = YOLOModel(
        base_channels=64,
        base_depth=2,
        n_classes=80
    ).to(device)
    logging.info("YOLOModel initialized")

    # 3. Debug trainer
    trainer = DebugTrainer(model)
    logging.info("DebugTrainer initialized")

    # 4. Train 100 steps
    logging.info("Starting debug training...")
    for step in range(100):
        loss = trainer.train_step(image, target)
        if step % 50 == 0:
            msg = f"[{step}] loss = {loss:.4f}"
            print(msg)
            logging.info(msg)
    
    logging.info("Debug training completed")

if __name__ == "__main__":
    main()
