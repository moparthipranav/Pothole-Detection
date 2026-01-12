import cv2
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_pil_image  # Remove it when in production
import torchvision.transforms as T
import numpy as np

class RoboFlowDataset(Dataset):

    def __init__(self, img_dir, label_dir, img_size):
        super().__init__()
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(
            self.label_dir,
            img_name.replace(".jpg", ".txt")
        )

        # Load the image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
    
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        targets = []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, cx, cy, bw, bh = map(float, line.split())

                    targets.append([cls, cx, cy, bw, bh])

        targets = torch.tensor(targets) if len(targets) else torch.zeros((0, 5))

        return image, targets
    
# Testing the data_ingestion layer
if __name__ == "__main__":
    dataset_loader = RoboFlowDataset(
        img_dir="C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/images",
        label_dir="C:/Users/Pranav/Downloads/Pothole Detection.v1i.yolov8/train/labels",
        img_size = 400
    )

    img, targets = dataset_loader[0]
    print("Image shape:", img.shape)
    print("Targets:", targets)


    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    h, w = img_np.shape[:2]

    for box in targets:
        cls, cx, cy, bw, bh = box.tolist()

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("gt_check", img_np)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
