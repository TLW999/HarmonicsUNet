import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention1D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttention1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        return x * (avg_out + max_out).unsqueeze(2)


class SpatialAttention1D(nn.Module):
    def __init__(self, num_channels):
        super(SpatialAttention1D, self).__init__()
        self.conv1 = nn.Conv1d(2, num_channels, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x * self.sigmoid(x)


class CBAM1D(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(CBAM1D, self).__init__()
        self.channel_attention = ChannelAttention1D(num_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention1D(num_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


if __name__ == '__main__':
    input_tensor = torch.randn(10, 33, 300)  # Batch size: 10, Channels: 64, Sequence length: 300
    model = CBAM1D(num_channels=33)
    output = model(input_tensor)
    print(output.shape)
