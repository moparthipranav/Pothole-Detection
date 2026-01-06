import torch
import torch.nn as nn
from src.model.C2fModule import Conv

class LocalConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SpatialAttention(nn.Module):
    '''
    The convolution are good at localized learning but fail to learn the dependencies within the same input.
    A pothole is only characterized as a pothole when it is surrounded by asphalt and cracks. In this situation, the information about the 
    edges and the texture is not usable for prediction, instead if we measure how much one pixel depends on another pixel or a specific localized 
    are is dependent on other localized areas of the picture, then our accuracy of measurement increases. This is enabled by spatial attention.
    '''

    def __init__(self, c):
        super().__init__()

        self.q = nn.Conv2d(c, c // 8, kernel_size=1) # Increasing the feature semantics and reducing the size for easier calculations
        self.k = nn.Conv2d(c, c // 8, kernel_size=1)
        self.v = nn.Conv2d(c, c, kernel_size=1)

        self.scale = (c // 8) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x).view(B, -1, H*W)
        k = self.k(x).view(B, -1, H * W)          # (B, C', HW)
        v = self.v(x).view(B, C, H * W)           # (B, C, HW)

        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * self.scale,
            dim=-1
        )

        out = torch.bmm(v, attn.transpose(1, 2))
        return out.view(B, C, H, W)

class ACmix(nn.Module):
    """Simplified ACmix-style block integrating Attention and Conv"""
    def __init__(self, c):
        super().__init__()

        self.local = LocalConv(c)
        self.attn = SpatialAttention(c)

        self.alpha = nn.Parameter(torch.ones(1, c, 1, 1))
    
    def forward(self, x):
        local_feat = self.local(x)
        attn_feat = self.attn(x)

        alpha = torch.sigmoid(self.alpha)  # constrain to (0,1)

        return alpha * local_feat + (1 - alpha) * attn_feat
    
class C2f_ACmix(nn.Module):
    """C2f module using ACmix instead of standard Bottlenecks"""
    def __init__(self, c1, c2, n=1):
        super().__init__()

        self.hidden = c2 // 2

        self.cv1 = Conv(c1, 2 * self.hidden, k=1)

        self.m = nn.ModuleList(
            ACmix(self.hidden) for _ in range(n)
        )

        self.cv2 = Conv((n + 2) * self.hidden, c2, k=1)
    
    def forward(self, x):
        x = self.cv1(x)

        # Channel split
        x1, x2 = x.chunk(2, dim=1)

        outputs = [x1, x2]

        for block in self.m:
            x2 = block(x2)
            outputs.append(x2)

        out = torch.cat(outputs, dim=1)
        return self.cv2(out)
    
# if __name__ == "__main__":
#     x = torch.randn(1, 128, 20, 20)
#     model = C2f_ACmix(c1=128, c2=256, n=3)

#     y = model(x)
#     print(y.shape)
