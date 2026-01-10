from torch.utils.data import DataLoader
from src.components.data_ingestion import RoboFlowDataset

train_dataset = RoboFlowDataset(
    img_dir="C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/images",
    label_dir="C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/labels"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: tuple(zip(*x))
)
