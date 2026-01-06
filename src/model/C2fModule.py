import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None): #kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()

        if p is None:
            p = k // 2  # SAME padding for odd kernels

        self.conv = nn.Conv2d(
            c1, c2, kernel_size=k, stride=s,
            padding=p, groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BottleNeck(nn.Module):
    
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, k=1)
        self.cv2 = Conv(c, c, k=3)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y
        
class C2f(nn.Module):
    '''
    The core of YOLOv8 is the C2f module which is about controlled feature reuse and maximizes gradient and feature efficiency.

    It follows the CSPDarknet pattern in which the tensor is split into two parts across the channel dimension
    The approacch is used for better feature preservation and in the context of this project, the C2f module can help us to detect smaller potholes

    The Bottleneck layer consists of two conv layers

    Step-by-Step architecture breakdown:
    1. Initial 1 x 1 convolution (Channel preparation)
    2. Channel Split
    3. Sequential BottleNeck Blocks (each bottleneck uses residual layers)
    4. Feature concatenation
    5. Final 1 x 1 filter convolution for depth integration

    Conv -> Split -> BottleNeck -> ... -> Concat -> Conv
    '''

    def __init__(self, c1, c2, n=1, shortcut=True):
        '''
        Docstring for __init__
        
        :param c1: Number of input channels
        :param c2: Number of output channels
        :param n: Number of bottleneck layers

        '''
        super().__init__()

        self.hidden_channels = c2 // 2

        #Initial Projection
        self.cv1 = Conv(c1, 2 * self.hidden_channels, k=1)

        # Sequential Bottleneck layers
        self.m = nn.ModuleList(
            BottleNeck(self.hidden_channels, shortcut) for _ in range(n)
        )

        # Final fusion layer
        self.cv2 = Conv((2 + n) * self.hidden_channels, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)

        # split channels
        x1, x2 = x.chunk(2, dim=1)

        outputs = [x1, x2]

        for block in self.m:
            x2 = block(x2)
            outputs.append(x2)

        return self.cv2(torch.cat(outputs, dim=1))

# Quick Sanity Tests

if __name__ == "__main__":
    x = torch.randn(1, 64, 80, 80)
    model = C2f(c1=64, c2=128, n=3)

    y = model(x)
    print(y.shape)