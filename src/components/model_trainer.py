import torch
import torch.nn as nn

from src.model.YoloModel import YOLOModel

class ModelTrainer:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = YOLOModel(
            base_channels=config["base_channels"],
            base_depth=config["base_depth"],
            n_classes=config["n_classes"]
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["lr"]
        )

        self.criterion = nn.MSELoss()
    
    def train_step(self, images, targets):
        images = images.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(images)

        # dummy loss (replace later)
        loss = self.criterion(outputs[0], targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()