import torch
from src.model.YoloModel import YOLOModel
from src.components.DataLoader import RoboFlowDataset
from src.components.DebugTrainer import DebugTrainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load ONE image
    dataset = RoboFlowDataset(
        img_dir="C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/images",
        label_dir="C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/labels"
    )

    image, target = dataset[0]
    image = image.unsqueeze(0).to(device)
    target = target.to(device)
    # 2. Model
    model = YOLOModel(
        base_channels=64,
        base_depth=2,
        n_classes=80
    ).to(device)

    # 3. Debug trainer
    trainer = DebugTrainer(model)

    # 4. Train 500 steps
    for step in range(100):
        loss = trainer.train_step(image, target)
        if step % 50 == 0:
            print(f"[{step}] loss = {loss:.4f}")

if __name__ == "__main__":
    main()
