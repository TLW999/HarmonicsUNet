import torch
import torch.nn as nn
from dsc import DSC,IDSC
import torch.nn.functional as F
class SFEBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(SFEBlock, self).__init__()

        # 主分支 - 两个卷积层
        self.conv1 = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1)  # 保证输入输出的特征图大小一致
        )
        self.bn1 = nn.BatchNorm2d(c_out)
        self.elu1 = nn.ELU()

        self.conv2 = nn.Conv2d(
            in_channels=c_out,
            out_channels=c_out,
            kernel_size=(3, 3),
            stride=1,
            padding=(1, 1)
        )
        self.bn2 = nn.BatchNorm2d(c_out)

        # 残差分支 - 用于调整输入的通道数
        self.residual_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,  # 使用 1x1 卷积调整通道数
            stride=1,
            padding=0
        )
        self.residual_bn = nn.BatchNorm2d(c_out)

        # 最大池化和 Dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 残差分支
        residual = self.residual_bn(self.residual_conv(x))  # 通过1x1卷积调整输入形状

        # 主分支 - 两个卷积模块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # 残差连接：输入残差与卷积输出相加
        x = x + residual

        # 最大池化和 Dropout
        x = self.pool(x)
        x = self.dropout(x)

        return x

# PGN 处理频谱图的长期依赖
class PGN(nn.Module):
    def __init__(self,c_out, kernel_size,dilation=2):
        super(PGN, self).__init__()
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

        self.gate =nn.Conv2d(
            in_channels=2 * c_out,
            out_channels=2 * c_out,
            kernel_size=(1, kernel_size),
            padding=(0, (kernel_size // 2)*dilation),
            dilation = dilation,
            )
        #注意力机制
        #self.attention=Attention(c_out)

    def gated_unit(self, x, hidden):
        x = torch.cat([x, hidden], dim=1)
        gate_out = self.gate(x)
        sigmoid_gate, tanh_gate = torch.chunk(gate_out, 2, dim=1)
        sigmoid_gate = torch.sigmoid(sigmoid_gate)
        tanh_gate = torch.tanh(tanh_gate)
        hidden = hidden * sigmoid_gate + (1 - sigmoid_gate) * tanh_gate
        return hidden

    def forward(self, x):
        x = self.conv(x)
        hidden = self.hidden_MLP(x)
        hidden = hidden + x
        hidden = self.gated_unit(x, hidden)
        # 自注意力机制
        # hidden = self.attention(hidden)
        return hidden

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
        self.pool = nn.MaxPool2d(kernel_size=(1, 2), stride=1, padding=(0, 1))

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

        self.long_term = PGN(c_out, kernel_size)

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

class FFMWithAttention(nn.Module):
    def __init__(self, dim1, dim2):
        super(FFMWithAttention, self).__init__()
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2 * 2, dim2, 1)

        self.fusion = nn.Sequential(IDSC(dim2 * 4, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    DSC(dim2, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    nn.Conv2d(dim2, dim2, 1),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU())

    def forward(self, x, y):
        #x是spctral_features[128,512,16,4]
        #y是gru_features[128,66,256]
        b, c, h, w = x.shape
        B, N, C = y.shape
        H=h
        W=w
        x = self.trans_c(x)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8,
                                                                                                         16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)

        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        out = torch.cat([x, y, out1, out2], dim=1)

        out = self.fusion(out)
        return out

class BirdSoundRecognitionWithBiGRUAndTPGN(nn.Module):
    def __init__(self, num_mel_bands, num_classes, in_channels=1, gru_hidden_size=128, gru_layers=1, c_out=128):
        super(BirdSoundRecognitionWithBiGRUAndTPGN, self).__init__()
        self.num_classes = num_classes
        self.c_out = c_out
        self.gru_hidden_size = gru_hidden_size
        self.gru_layers = gru_layers
        #频率特征提取之前对频谱图进行预处理
        self.sconv = nn.Sequential(
            nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1),
        nn.BatchNorm2d(64), # 批量归一化
        nn.ELU(),  # ELU 激活函数
        )
        # TPGN 用于进一步处理卷积特征，处理频谱图中的时域和频域信息
        self.tpgn = TPGN(c_in=64, c_out=c_out, kernel_size=7, period=3)
        # BiGRU 用于时间序列建模
        self.gru = nn.GRU(
            input_size=126 * num_mel_bands,  # 压缩后的频率维度 * 通道数
            hidden_size=gru_hidden_size,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True
        )
        # 频谱特征处理
        # SFEBlock 叠加层
        self.block1 = SFEBlock(64, 128)
        self.block2 = SFEBlock(128, 256)
        self.block3 = SFEBlock(256, 512)
        #注意力机制
        self.attention = FFMWithAttention(512, 256)
        # 最终分类层
        # 全局池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 分类层
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        batch_size, _, _, time_frames = x.shape

        original_spectral= self.sconv(x)

        #卷积层来提取频谱特征
        spectral_features = self.block1(original_spectral)
        spectral_features = self.block2(spectral_features)
        spectral_features = self.block3(spectral_features)
        # TPGN 处理卷积特征
        tpgn_features = self.tpgn(x)
        b, c, f, t = tpgn_features.shape
        tpgn_features = tpgn_features.permute(0, 3, 1, 2)  # reshape 为 (batch_size, time_steps, feature_dim)
        tpgn_features = tpgn_features.reshape(batch_size, t, -1)
        # 时间序列建模
        tpgn_features, _ = self.gru(tpgn_features)
        gru_features = F.interpolate(tpgn_features.permute(0, 2, 1), size=64, mode='linear', align_corners=False)
        gru_features = gru_features.permute(0, 2, 1)
        out=self.attention(spectral_features, gru_features)
        # 全局池化层，减少维度
        out = self.global_pool(out)  # 结果形状： (batch_size, 256, 1, 1)

        # 展平为 (batch_size, 256)
        out = out.view(out.size(0), -1)

        # 分类层
        out = self.fc(out)
        return out



if __name__ == '__main__':
    model = BirdSoundRecognitionWithBiGRUAndTPGN(num_mel_bands=128, num_classes=10)

    x = torch.randn(128, 1, 128, 32)  # 示例输入
    output = model(x)
    print(output.shape)