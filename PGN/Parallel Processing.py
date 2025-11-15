import torch.nn as nn
import torch

class TestDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=3,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=(0,0)
        )

    def forward(self, x):
        output = self.deconv(x)
        print("Input Size:", x.shape[2:])  # 假设输入尺寸为 62x62
        print("Output Size:", output.shape[2:])  # 应输出 125x125
        return output


# 测试
x = torch.randn(1, 3, 64, 62)
model = TestDeconv()
output = model(x)