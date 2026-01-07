import torch
from src.model.YoloModel import YOLOModel
from src.model.tests.BackboneTesting import image_to_matrix
import matplotlib.pyplot as plt

def show_tensor_plt(tensor):
    img = tensor.squeeze(0).cpu()

    img = img.permute(1,2,0).numpy()

    plt.imshow(img)
    plt.axis('off')
    plt.show()

def test_full_forward():
    model = YOLOModel(
        base_channels=64,
        base_depth=2,
        n_classes=80
    )

    # x = torch.randn(1, 3, 640, 640)
    my_input = image_to_matrix('src/model/tests/Test.jpg', size=640)
    show_tensor_plt(my_input)

    outputs = model(my_input)

if __name__ == "__main__":
    test_full_forward()
