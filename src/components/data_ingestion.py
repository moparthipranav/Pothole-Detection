import cv2
import os
import torch
from torch.utils.data import Dataset

class RoboFlowDataset(Dataset):

    def __init__(self, img_dir, label_dir, img_size=640):
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

                    # Convert normalized → absolute
                    cx *= self.img_size
                    cy *= self.img_size
                    bw *= self.img_size
                    bh *= self.img_size

                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2

                    targets.append([cls, x1, y1, x2, y2])

        targets = torch.tensor(targets) if len(targets) else torch.zeros((0, 5))

        return image, targets