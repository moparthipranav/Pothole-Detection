import torch
from src.model.YoloNeck import YoloNeckHead  # adjust import if needed


def test_neck_forward():
    base_channels = 64
    base_depth = 2
    n_classes = 80

    model = YoloNeckHead(
        base_channels=base_channels,
        base_depth=base_depth,
        n_classes=n_classes
    )

    # Fake backbone outputs
    p2 = torch.randn(1, base_channels * 2, 160, 160)
    p3 = torch.randn(1, base_channels * 4, 80, 80)
    p4 = torch.randn(1, base_channels * 8, 40, 40)
    p5 = torch.randn(1, base_channels * 16, 20, 20)

    outputs = model(p2, p3, p4, p5)

    for i, out in enumerate(outputs, start=2):
        print(f"P{i} output shape:", out.shape)

if __name__ == "__main__":
    test_neck_forward()
