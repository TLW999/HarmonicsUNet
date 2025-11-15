import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import warnings
warnings.filterwarnings('ignore')

#######################
# 核心模块定义
#######################
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ch_att = self.channel_att(x)
        x = x * ch_att
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sp_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * sp_att

class BirdTimeFreqUnit(nn.Module):
    def __init__(self, in_ch, out_ch,use_channel_boost=False):
        super().__init__()

        self.use_channel_boost = use_channel_boost

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch//4 , 3, padding=1),
            nn.BatchNorm2d(out_ch // 4),
            nn.GELU()
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, 5, padding=2),
            nn.BatchNorm2d(out_ch // 4),
            nn.GELU()
        )
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, 7, padding=3),  # 7x7 卷积核
            nn.BatchNorm2d(out_ch // 4),
            nn.GELU()
        )

        # 空间上下文融合
        self.spatial_fuse = nn.Sequential(
            nn.Conv2d((out_ch //4)*3, out_ch, 3, padding=1),  # 深度卷积减少参数
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
        #通道融合
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # 通道增强模块,只在编码器的最后一层使用
        if use_channel_boost:
            self.channel_boost = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_ch, out_ch // 8, 1),
                nn.GELU(),
                nn.Conv2d(out_ch // 8, out_ch, 1),
                nn.Sigmoid()  # 通道级门控
            )
        else:
            self.channel_boost = None

        self.channel_expand = nn.Conv2d(in_channels=out_ch//4, out_channels=32, kernel_size=1)
        self.channel_expand1 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=1)
        # 残差路径
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        # 多尺度特征提取
        feat_3x3 = self.conv_3x3(x)
        feat_5x5 = self.conv_5x5(x)
        feat_7x7 = self.conv_7x7(x)

        # 拼接多尺度特征
        fused = torch.cat([feat_3x3, feat_5x5, feat_7x7], dim=1)  # [B, out_ch, H, W]

        # 空间上下文融合
        fused = self.spatial_fuse(fused)

        # 通道融合
        base = self.fuse_conv(fused)

        # 通道增强（仅在启用时应用）
        if self.channel_boost is not None:
            channel_weights = self.channel_boost(base)
            boosted = base * channel_weights
        else:
            boosted = base

        return F.gelu(boosted + residual)


class DynamicJointEncoding(nn.Module):
    """动态时频联合位置编码，根据输入特征生成编码参数"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 动态参数生成网络
        self.param_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 提取全局特征
            nn.Flatten(),
            nn.Linear(dim, 4 * dim + 1),  # 生成time/freq/cross参数和log_base
            nn.GELU()  # 引入非线性
        )

        # 初始化参数生成器的权重
        nn.init.normal_(self.param_generator[2].weight, mean=0, std=0.02)
        # 融合模块（将 C + D → C）
        self.fusion_conv = nn.Conv2d(dim + dim, dim, kernel_size=1)

    def _get_position_grid(self, F, T, device):
        """生成归一化的时频坐标网格"""
        f_coords = torch.linspace(0, 1, F, device=device)
        t_coords = torch.linspace(0, 1, T, device=device)
        return torch.stack(torch.meshgrid(f_coords, t_coords, indexing='ij'), dim=-1)  # [F, T, 2]

    def forward(self, x):
        B, C, F, T = x.shape
        device = x.device

        # 生成动态参数 --------------------------------------------------
        params = self.param_generator(x)  # [B, 4*dim+1]
        # 分解参数项
        time_factors = params[:, :self.dim]  # [B, D]
        freq_factors = params[:, self.dim:2 * self.dim]  # [B, D]
        cross_weights = params[:, 2 * self.dim:4 * self.dim].view(B, self.dim, 2)  # [B, D, 2]
        log_base = params[:, -1]  # [B]

        # 生成坐标网格 -------------------------------------------------
        grid = self._get_position_grid(F, T, device)  # [F, T, 2]

        # 计算基础编码 -------------------------------------------------
        safe_base = torch.exp(torch.clamp(log_base, max=10)).unsqueeze(-1) + 1e-6 # [B, 1]
        exponents = torch.arange(self.dim // 2, device=device).float()
        inv_freq = 1.0 / (safe_base ** (exponents / (self.dim // 2 - 1)))
        # 交叉项编码
        raw_enc = torch.einsum('ftd,bcd->bftc', grid, cross_weights)  # [B, F, T, D]

        # 频域调制
        sin_part = torch.sin(raw_enc[..., ::2] * inv_freq.unsqueeze(1).unsqueeze(1))
        cos_part = torch.cos(raw_enc[..., 1::2] * inv_freq.unsqueeze(1).unsqueeze(1))

        # 合成位置编码
        pos_enc = torch.zeros_like(raw_enc)
        pos_enc[..., 0::2] = sin_part
        pos_enc[..., 1::2] = cos_part

        # 时空调制 ------------------------------------------------
        # 时间调制 (B, T, D) -> (B, 1, T, D)
        time_mod = torch.arange(T, device=device).view(1, T, 1) * time_factors.unsqueeze(1)
        time_mod = time_mod.view(B, 1, T, self.dim)

        # 频率调制 (B, F, D) -> (B, F, 1, D)
        freq_mod = torch.arange(F, device=device).view(1, F, 1) * freq_factors.unsqueeze(1)
        freq_mod = freq_mod.view(B, F, 1, self.dim)

        # 综合编码 ------------------------------------------------
        modulated_enc = pos_enc + time_mod + freq_mod  # [B, F, T, D]
        modulated_enc = modulated_enc.permute(0, 3, 1, 2)  # [B, D, F, T]

        # 拼接再映射
        x_cat = torch.cat([x, modulated_enc], dim=1)  # [B, C+D, F, T]
        x_fused = self.fusion_conv(x_cat)  # [B, C, F, T]
        # 维度调整并残差连接
        return x_fused


class TimeFreqGlobalBlock(nn.Module):
    def __init__(self, dim, heads=8, reduction_ratio=4):
        super().__init__()
        # 位置编码
        self.dim = dim
        self.pos_enc = DynamicJointEncoding(dim)

        # 全局分解与重构模块
        self.global_decomp = nn.ModuleList([
            nn.Sequential(  # 时间全局路径
                nn.AdaptiveAvgPool2d((1, None)),
                nn.Conv2d(dim, dim // reduction_ratio, 1),
                nn.GELU(),
                nn.Conv2d(dim // reduction_ratio, dim, 1)
            ), 
            nn.Sequential(  # 频率全局路径
                nn.AdaptiveAvgPool2d((None, 1)),
                nn.Conv2d(dim, dim // reduction_ratio, 1),
                nn.GELU(),
                nn.Conv2d(dim // reduction_ratio, dim, 1)
            )
        ])
        #通道注意力
        self.DirectionalFusionGate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 2C, 1, 1]
            nn.Conv2d(dim * 2, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim * 2, 1),
            nn.Softmax(dim=1)  # 逐通道归一化
        )
        #提升非线性处理
        self.reconstruct_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        # 全局动态注意力
        self.global_attn = nn.MultiheadAttention(dim, heads, batch_first=True)

        # 全局上下文调制
        self.global_modulator = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim, 1),
            nn.Sigmoid()
        )

        # 轻量化局部特征提取
        self.local_extract =nn.Sequential(
                nn.Conv2d(dim, dim, 5, padding=2),
                nn.BatchNorm2d(dim),
                nn.GELU()
            )

        # 全局引导的局部门控
        self.local_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # 最终融合
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, F, T = x.shape

        # 1. 注入动态时频位置编码
        x = self.pos_enc(x)  # [B, C, F, T]

        # 2. 全局分解与重构
        time_global = self.global_decomp[0](x)  # [B, C, 1, T]
        freq_global = self.global_decomp[1](x)  # [B, C, F, 1]
        time_global = time_global.expand(-1, -1, F, -1)  # [B, C, F, T]
        freq_global = freq_global.expand(-1, -1, -1, T)  # [B, C, F, T]
        combined = torch.cat([time_global, freq_global], dim=1)  # [B, 2C, F, T]
        weights = self.DirectionalFusionGate(combined)  # [B, 2C, 1, 1]
        time_weights = weights[:, :self.dim]  # [B, C, 1, 1]
        freq_weights = weights[:, self.dim:]  # [B, C, 1, 1]
        global_reconstructed = combined[:, :self.dim] * time_weights + combined[:, self.dim:] * freq_weights
        global_reconstructed = self.reconstruct_refine(global_reconstructed)

        # 3. 全局动态注意力
        global_seq = rearrange(global_reconstructed, 'b c f t -> b (f t) c')  # [B, F*T, C]
        global_attn_out, _ = self.global_attn(global_seq, global_seq, global_seq)
        global_attn_out = rearrange(global_attn_out, 'b (f t) c -> b c f t', f=F, t=T)

        # 4. 全局上下文调制
        modulator = self.global_modulator(global_attn_out)
        global_feat = global_attn_out * modulator

        # 5. 轻量化局部特征提取
        local_feat = self.local_extract(x)

        # 6. 全局引导的局部门控
        local_weight = self.local_gate(global_feat)
        gated_local = local_feat * local_weight

        # 7. 全局与局部融合
        combined = torch.cat([global_feat, gated_local], dim=1)  # [B, 2C, F, T]
        out = self.fusion_conv(combined)
        out = self.norm(out)
        return out



#金字塔特征融合
class FusionPyramidUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()
        # 多尺度提取
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch // 4, kernel_size=5, stride=(1,1), padding=2),
                nn.BatchNorm2d(ch // 4),  # 归一化
                nn.GELU()
            ),  # 低频
            nn.Sequential(
                nn.Conv2d(ch, ch // 4, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch // 4),  # 归一化
                nn.GELU()
            ),  # 中频
            nn.Sequential(
                nn.Conv2d(ch, ch - (ch // 4) * 2, kernel_size=3, dilation=2, padding=2),
                nn.BatchNorm2d(ch - (ch // 4) * 2),  # 归一化
                nn.GELU()
            )  # 高频
        ])


        # 动态注意力
        self.att = CBAMBlock(ch)

    def forward(self, x):
        # 多尺度特征提取
        x_low = F.avg_pool2d(x, kernel_size=(2, 1))
        low = self.convs[0](x_low)
        mid = self.convs[1](x)
        x_high = F.max_pool2d(x, kernel_size=(2, 1))
        high = self.convs[2](x_high)

        # 上采样对齐维度
        low = F.interpolate(low, x.shape[-2:], mode='nearest')
        high = F.interpolate(high, x.shape[-2:], mode='nearest')

        # 归一化处理，确保拼接后数值不会过大
        fused = torch.cat([low, mid, high], dim=1)

        # 使用 Sigmoid 约束注意力权重
        out =  fused * torch.sigmoid(self.att(fused))
        return out


#谐波增强单元
class HarmonicEnhanceUnit(nn.Module):
    """谐波增强单元"""

    def __init__(self, ch):
        super().__init__()
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch // 4, 3, dilation=d, padding=d, groups=ch // 8),
                nn.BatchNorm2d(ch // 4),
                nn.GELU()
            ) for d in [1, 3, 5]
        ])

        # 动态置信度检测
        self.confidence = nn.Sequential(
            nn.Conv2d(ch, ch // 8, 1),
            nn.GELU(),
            nn.Conv2d(ch // 8, len(self.dilated_convs), 1),  # 输出 3 个权重，对应 3 个分支
            nn.Softmax(dim=1)
        )

        self.conv = nn.Conv2d(ch // 4, ch, 1)

        #谐波补偿（使用 Sigmoid）
        self.harmonic_compensator = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):
        # 提取谐波特征
        harmonics = [conv(x) for conv in self.dilated_convs]  # 列表，3 个 [B, ch//4, H, W]

        # 动态置信度检测
        weights = self.confidence(x)  # [B, 3, H, W]
        weights = weights.unsqueeze(2)  # [B, 3, 1, H, W]，为广播准备

        # 加权融合谐波特征
        harmonics_stack = torch.stack(harmonics, dim=1)  # [B, 3, ch//4, H, W]
        weighted = (harmonics_stack * weights).sum(dim=1)  # [B, ch//4, H, W]
        weighted = self.dropout(weighted)  # 在融合后应用 dropout

        # 恢复通道数
        weighted = self.conv(weighted)  # [B, ch, H, W]

        # 谐波补偿门控
        gate = self.harmonic_compensator(x)  # [B, ch, H, W]
        out = gate * x + (1 - gate) * weighted  # 加权融合原始特征和增强特征

        return out


#时频全局模块
class TemporalBandGate(nn.Module):
    """时频联合动态模块"""

    def __init__(self, ch):
        super().__init__()
        # 时频联合特征提取
        self.context = nn.Sequential(
            nn.Conv2d(ch, ch // 4, 3, padding=1),  # 提取局部时频上下文
            nn.GELU(),
            nn.Conv2d(ch // 4, ch, 1)  # 压缩并恢复通道
        )

        # 动态融合权重生成
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch, 1),  # 1x1卷积生成逐通道权重
            nn.Sigmoid()  # 生成[0,1]范围的加权图
        )


    def forward(self, x):
        # 提取时频联合特征
        context_feat = self.context(x)  # [B, C, F, T]

        # 生成动态加权图
        weights = self.gate(context_feat)  # [B, C, F, T]

        # 加权特征
        gated_feat = x * weights  # [B, C, F, T]

        return gated_feat

class SemanticFusionBlock(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
        # 频带金字塔特征融合
        self.freq_pyramid = FusionPyramidUnit(out_ch)
        # 谐波增强单元
        self.harmonic = HarmonicEnhanceUnit(out_ch)
        #时频全局模块
        self.temporal = TemporalBandGate(out_ch)
        #残差融合
        self.fusion_weights = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch // 4, 1),
            nn.GELU(),
            nn.Conv2d(out_ch // 4, 3, 1),
            nn.Softmax(dim=1)
        )
        # 特征精炼
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),  # 先调整通道
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1),  # 细化局部特征
            nn.BatchNorm2d(out_ch)  # 避免梯度不稳定
        )
    def forward(self, fused):
        # 多尺度处理
        pyramid_feat = self.freq_pyramid(fused)
        # 谐波结构增强
        harmonic_feat = self.harmonic(pyramid_feat)
        # 时序连续性建模
        temporal_feat = self.temporal(harmonic_feat)
        # 融合多模块特征
        feat_stack = torch.cat([pyramid_feat, harmonic_feat, temporal_feat], dim=1)  # [B, 3*out_ch, F, T]
        weights = self.fusion_weights(feat_stack)  # [B, 3, F, T]
        fused_feat = (
                pyramid_feat * weights[:, 0:1] +
                harmonic_feat * weights[:, 1:2] +
                temporal_feat * weights[:, 2:3]
        )
        return self.refine(fused_feat)


#######################
# U-Net主干网络
#######################
#解码器
class AdaptiveFusionDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # 调整通道数
            nn.ReLU(),
        )
        # 保留原始 x 和 skip 全部信息
        # self.reduce = nn.Sequential(nn.Conv2d(out_ch * 2, out_ch, 1),)

        # # 融合
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch // 4, 2, 1),
            nn.Softmax(dim=1)  # 产生两个归一化的权重图
        )

        # 融合进一步处理
        self.fusion = SemanticFusionBlock(out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        #第二种融合方法
        # 动态融合
        concat_feat = torch.cat([x, skip], dim=1)
        gate_weights = self.fusion_gate(concat_feat)
        out = x * gate_weights[:, 0:1] + skip * gate_weights[:, 1:2]

        # fused = torch.cat([x, skip], dim=1)
        # fused = self.reduce(fused)
        out = self.fusion(out)
        return out

#分类头
class Classifier(nn.Module):
    def __init__(self, in_ch, num_classes, time_steps=32):
        super().__init__()
        self.pre_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.GELU()
        )
        self.pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, None)),
            nn.AdaptiveMaxPool2d((1, None))
        ])
        self.norm = nn.LayerNorm(in_ch * time_steps * 2)
        self.fc = nn.Sequential(
            nn.Linear(in_ch * time_steps * 2, 256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.pre_conv(x)
        avg_pool = self.pool[0](x)
        max_pool = self.pool[1](x)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = Rearrange('b c 1 t -> b (c t)')(x)
        x = self.norm(x)
        return self.fc(x)

class UNetSpectroFusion(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # 编码器
        self.encoder_block = nn.ModuleList()
        # 第一块编码器
        self.encoder_block.append(nn.Sequential(
            BirdTimeFreqUnit(1,32,use_channel_boost=False),
            nn.AvgPool2d(kernel_size=(2, 1))  # 仅在频率轴下采样
            )
        )
        #第二块编码器
        self.encoder_block.append(nn.Sequential(
            BirdTimeFreqUnit(32, 64,use_channel_boost=False),
            nn.AvgPool2d(kernel_size=(2, 1))  # 仅在频率轴下采样
            )
        )
        #第三块编码器
        self.encoder_block.append(nn.Sequential(
            BirdTimeFreqUnit(64, 128,use_channel_boost=False),
            nn.AvgPool2d(kernel_size=(2, 1))
            )
        )
        #第四块编码器
        self.encoder_block.append(nn.Sequential(
            BirdTimeFreqUnit(128, 256,use_channel_boost=True),
         )
        )

        #瓶颈层
        self.fusions = TimeFreqGlobalBlock(dim=256)

        # 解码器（进行底层语义与高层语义的融合）
        self.decoder = nn.ModuleList([
            AdaptiveFusionDecoder(256, 128,scale_factor=(1,1)),
            AdaptiveFusionDecoder(128, 64,scale_factor=(2,1)),
            AdaptiveFusionDecoder(64, 32,scale_factor=(2,1)),
        ])

        # 分类头
        self.classifier = Classifier(32, num_classes)


    def forward(self, x):
        # 输入x: [B, C, F, T]
        skips = []

        # 编码器
        for block in self.encoder_block:
            x = block(x)
            skips.append(x)
        # 瓶颈层
        x = self.fusions(x)
        # 解码器
        for i, dec in enumerate(self.decoder):
            x = dec(x, skips[-(i + 2)])
        #分类
        out = self.classifier(x)
        return out


import time
if __name__ == '__main__':
    model = UNetSpectroFusion(num_classes=40)
    start_time = time.time()  # 记录开始时间
    x = torch.randn(128, 1, 128, 32)  # Mel频谱图示例
    output = model(x)
    print(output.shape)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"代码运行耗时：{elapsed_time:.6f} 秒")