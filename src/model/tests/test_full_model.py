import torch
from src.model.YoloModel import YOLOModel

def test_full_forward():
    model = YOLOModel(
        base_channels=64,
        base_depth=2,
        n_classes=80
    )

    x = torch.randn(1, 3, 640, 640)
    outputs = model(x)

    for i, out in enumerate(outputs, start=2):
        print(f"P{i} output shape:", out.shape)

if __name__ == "__main__":
    test_full_forward()
