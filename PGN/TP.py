import torch
import torch.nn as nn
import torch.nn.functional as F

class PGN_2d(nn.Module):
    def __init__(self,c_out, kernel_size,dilation=2):
        super(PGN_2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=kernel_size),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),

        )
        self.hidden_MLP = nn.Conv2d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=(1,kernel_size),
            stride=1,
            padding=(0,(kernel_size // 2)*dilation),
            dilation = dilation,
        )

        self.hidden_MLP2 = nn.Conv2d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=(1, kernel_size*2),
            stride=1,
            padding=(0, ((kernel_size*2) // 2) * dilation),
            dilation=dilation,
        )

        self.gate =nn.Conv2d(
            in_channels=4 * c_out,
            out_channels=2 * c_out,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size // 2)*dilation),
            dilation = dilation,
            )
        #注意力机制
        self.attention=lChannelAttention(c_out)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.adjust_hidden = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        # 时间池化操作
        self.time_pool = nn.MaxPool2d(kernel_size=(1, kernel_size), stride=1, padding=(0, kernel_size // 2))

    def gated_unit(self, x, hidden):
        x = torch.cat([x, hidden], dim=1)
        gate_out = self.gate(x)
        sigmoid_gate, tanh_gate = torch.chunk(gate_out, 2, dim=1)
        sigmoid_gate = torch.sigmoid(sigmoid_gate)
        tanh_gate = torch.tanh(tanh_gate)
        hidden = self.adjust_hidden(hidden)
        hidden = hidden * sigmoid_gate + (1 - sigmoid_gate) * tanh_gate
        return hidden

    def forward(self, x):
        x = self.conv(x)
        hidden = self.hidden_MLP(x)
        hidden2 = self.hidden_MLP2(x)
        hidden2 = F.interpolate(hidden2, size=(hidden2.size(2), hidden.size(3)), mode='bilinear', align_corners=False)  # 上采样
        hidden = torch.cat([hidden, hidden2], dim=1)
        # 使用零填充扩展通道维度
        x=self.conv1(x)
        hidden = hidden + x
        hidden = self.gated_unit(x, hidden)
        # 进行时间池化
        hidden = self.time_pool(hidden)
        # 自注意力机制
        hidden = self.attention(hidden)
        return hidden


class lChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(lChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用全局平均池化来生成通道注意力
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)
        pooled = avg_pool + max_pool
        attention = self.fc2(F.relu(self.fc1(pooled)))
        return x * self.sigmoid(attention)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, 1)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, 1)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)  # 平均池化
        max_pool = torch.amax(x, dim=(2, 3), keepdim=True)  # 最大池化
        x = avg_pool + max_pool
        x = self.fc2(F.relu(self.fc1(x)))
        return torch.sigmoid(x) * x


class short_term_deal(nn.Module):
    def __init__(self, c_in, c_out, period,dilation=2):
        super(short_term_deal, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=period),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        # 使用多尺度卷积
        self.fc1 = nn.Conv2d(in_channels=c_in,
                             out_channels=c_out,
                             kernel_size=(1, period),
                             stride=1,
                             padding=(0, (period // 2)*dilation),
                             dilation = dilation,)
        self.fc2 = nn.Conv2d(in_channels=c_in,
                             out_channels=c_out,
                             kernel_size=(1, period * 2),
                             stride=1,
                             padding=(0, ((period * 2) // 2)*dilation),
                             dilation = dilation,)

        # 最大池化
        self.pool = nn.MaxPool2d(kernel_size=(1, period),stride=1,padding=(0, period // 2))

        # 通道注意力机制
        self.channel_attention = ChannelAttention(2*c_out)

    def forward(self, x):
        x=self.conv(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x1 = F.interpolate(x1, size=(x1.size(2),x2.size(3)), mode='bilinear', align_corners=False)  # 上采样
        x = torch.cat([x1, x2], dim=1)
        x = self.pool(x)  # 池化层
        x = self.channel_attention(x) * x  # 注意力机制
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 全局平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

# TPGN 模块
class TPGN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, period):
        #c_in=64,c_out=128
        super(TPGN, self).__init__()

        self.long_term = PGN_2d(c_out, kernel_size)

        self.short_term = short_term_deal(c_in, c_out, period)
        self.fc = nn.Conv2d(
            in_channels=2 * c_out+c_out,
            out_channels=c_out,
            kernel_size=1
        )
        #注意力机制
        self.attention = SpatialAttention(kernel_size=kernel_size)
    def forward(self, x):
        long_term_out = self.long_term(x)
        short_term_out = self.short_term(x)
        long_term_out=F.interpolate(long_term_out,size=(short_term_out.size()[2:]), mode='bilinear', align_corners=False)
        out = torch.cat([long_term_out, short_term_out], dim=1)
        out=self.attention(out)
        out = self.fc(out)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch_size, time_steps, hidden_dim)
        attention_scores = self.attention_weights(x).squeeze(-1)  # (batch_size, time_steps)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, time_steps)
        context = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_dim)
        return context, attention_weights



class BirdSoundRecognitionWithBiGRUAndTPGN(nn.Module):
    def __init__(self, num_mel_bands, num_classes, gru_hidden_size=128, gru_layers=1, tpgn_c_out=128):
        super(BirdSoundRecognitionWithBiGRUAndTPGN, self).__init__()
        self.num_classes = num_classes

        # TPGN 用于进一步处理卷积特征
        self.tpgn = TPGN(c_in=64, c_out=tpgn_c_out, kernel_size=7, period=3)

        # BiGRU 用于时间序列建模
        self.gru = nn.GRU(
            input_size=tpgn_c_out *(num_mel_bands-2),  # 压缩后的频率维度 * 通道数
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )
        # 自注意力机制
        self.attention = AttentionLayer(gru_hidden_size*2)
        # 分类器
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size*2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, _, _, time_frames = x.shape
        # print(time_frames)
        # print("x shape:", x.shape)
        # TPGN 处理卷积特征
        x = self.tpgn(x)
        # 展平为 GRU 输入
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2)
        x=x.reshape(batch_size, t, -1)

        # 时间序列建模
        x, _ = self.gru(x)

        # 注意力机制
        features, attn_weights = self.attention(x)
        # 分类
        output = self.fc(features)

        return output




if __name__ == '__main__':
    model = BirdSoundRecognitionWithBiGRUAndTPGN(num_mel_bands=128, num_classes=10)

    x = torch.randn(128, 1, 128, 32)  # 示例输入
    output = model(x)
    print(output.shape)