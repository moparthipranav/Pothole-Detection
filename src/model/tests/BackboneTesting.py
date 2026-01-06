import matplotlib.pyplot as plt
import torch
from src.model.Backbone import YOLOBackbone
from PIL import Image
import torchvision.transforms as T

def image_to_matrix(image_path, size = 640):
    # 1. Load the image
    image = Image.open(image_path).convert('RGB')

    # 2. Defining the resizing pipeline
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor()
    ])

    # 3. Applying the transformations and adding a batch dimension (B, C, H, W)
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor

# ---- Running the test -----
my_input = image_to_matrix('src/model/tests/Test.jpg', size=640)

print(f"Matrix shape ready for backbone: {my_input.shape}")
def test_backbone(model, input_size=640):
    # 1. Create a dummy image (Batch size = 1, Channels = 3, H=640, W=640)
    # Using 1s or random values is fine for shape testing
    dummy_img = torch.randn(1, 3, input_size, input_size)
    print(f"Input Shape: {dummy_img.shape}\n" + "-"*30)

    # 2. Pass through Stem
    x = model.stem(dummy_img)
    print(f"After Stem:    {x.shape}")

    # 3. Pass through Stage 1
    x = model.stage1_conv(x)
    x = model.stage1_c2f(x)
    print(f"After Stage 1: {x.shape}")

    # 4. Pass through Stage 2 (P3)
    x = model.stage2_conv(x)
    p3 = model.stage2_c2f(x)
    print(f"After Stage 2 (P3): {p3.shape}")

    # Get the feature map from P3 (64 channels)
    # Let's look at the 10th channel
    feature_map = p3[0, 10, :, :].detach().numpy()

    plt.imshow(feature_map, cmap='viridis')
    plt.title("Visualizing P3 - Channel 10")
    plt.show()

    # 5. Pass through Stage 3 (P4)
    x = model.stage3_conv(p3)
    p4 = model.stage3_c2f(x)
    print(f"After Stage 3 (P4): {p4.shape}")

    # 6. Pass through Stage 4 (P5)
    x = model.stage4_conv(p4)
    p5 = model.stage4_c2f(x)
    print(f"After Stage 4 (P5): {p5.shape}")

    # 7. Final SPPF
    out = model.sppf(p5)
    print(f"After SPPF:    {out.shape}")

# Initialize model (Nano scale: base_channels=16, base_depth=1)
model = YOLOBackbone(base_channels=16, base_depth=1)
test_backbone(model)



