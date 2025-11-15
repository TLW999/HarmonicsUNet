import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from EfficientNet.EffNetV2 import SEBlock
import warnings
warnings.filterwarnings('ignore')

#######################
# 核心模块定义
#######################

class ChannelSpatialAttention(nn.Module):
    """二维注意力模块"""

    def __init__(self, in_ch):
        super().__init__()
        # 通道注意力的共享 MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch // 4, in_ch, 1)
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力：Avg + Max 都走 MLP
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        ch_att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        x = x * ch_att
        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sp_att = self.spatial_att(torch.cat([max_pool, avg_pool], dim=1))
        return x * sp_att

class BirdFeature(nn.Module):
    def __init__(self, in_ch, out_ch,use_channel_boost=False):
        super().__init__()
        self.use_channel_boost = use_channel_boost

        # 多尺度基础特征提取 (并行分支)
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, 3, padding=1),
            nn.BatchNorm2d(out_ch // 4),
            nn.GELU()
        )
        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch // 4, 5, padding=4, dilation=2),
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
            nn.Conv2d((out_ch // 4)* 3, out_ch, 3, padding=1),  # 深度卷积减少参数
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout(0.1)
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

        # 空间细节保留
        self.detail_retain = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch // 2),  # 深度卷积
            nn.BatchNorm2d(out_ch),
        )

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

        # 空间细节增强
        detailed = self.detail_retain(boosted)

        return F.gelu(detailed + residual)


class DynamicJointEncoding(nn.Module):
    """动态时频联合位置编码，根据输入特征生成编码参数"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # 动态参数生成网络（轻量级设计）
        self.param_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 提取全局特征
            nn.Flatten(),
            nn.Linear(dim, 4 * dim + 1),  # 生成time/freq/cross参数和log_base
            nn.GELU()  # 引入非线性
        )

        # 初始化参数生成器的权重
        nn.init.normal_(self.param_generator[2].weight, mean=0, std=0.02)

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
        inv_freq = 1.0 / (safe_base ** (exponents / (self.dim // 2 - 1)))  # 限制最大值
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

        # 维度调整并残差连接
        return x + modulated_enc.permute(0, 3, 1, 2)  # [B, C, F, T]


class SpectroFusionBlock(nn.Module):

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
        self.channel_fusion = nn.Sequential(
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

        # 轻量化局部特征提取（多尺度）
        self.local_extract = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim // 2, k, padding=k // 2, groups=dim // 4),
                nn.BatchNorm2d(dim // 2),
                nn.GELU()
            ) for k in [3, 5]
        ])
        self.local_fuse = nn.Conv2d(dim, dim, 1)

        # 全局引导的局部门控
        self.local_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # 最终融合
        self.fusion_conv = nn.Conv2d(dim * 2, dim, 1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        resdual = x
        B, C, F, T = x.shape

        # 1. 注入动态时频位置编码
        x = self.pos_enc(x)  # [B, C, F, T]

        # 2. 全局分解与重构
        time_global = self.global_decomp[0](x)  # [B, C, 1, T]
        freq_global = self.global_decomp[1](x)  # [B, C, F, 1]
        time_global = time_global.expand(-1, -1, F, -1)  # [B, C, F, T]
        freq_global = freq_global.expand(-1, -1, -1, T)  # [B, C, F, T]
        combined = torch.cat([time_global, freq_global], dim=1)  # [B, 2C, F, T]
        weights = self.channel_fusion(combined)  # [B, 2C, 1, 1]
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
        local_feats = [m(x) for m in self.local_extract]
        local_feat = torch.cat(local_feats, dim=1)
        local_feat = self.local_fuse(local_feat)

        # 6. 全局引导的局部门控
        local_weight = self.local_gate(global_feat)
        gated_local = local_feat * local_weight

        # 7. 全局与局部融合
        combined = torch.cat([global_feat, gated_local], dim=1)  # [B, 2C, F, T]
        out = self.fusion_conv(combined)
        out = self.norm(out)
        # 残差连接
        return out + resdual

#频带金字塔特征融合
class FrequencyPyramidUnit(nn.Module):
    def __init__(self, ch):
        super().__init__()

        # 动态分配通道数
        base_ch = ch // 3  # 基准通道数
        remainder = ch % 3  # 余数
        ch_low = base_ch
        ch_mid = base_ch
        ch_high = base_ch+ remainder  # 高频分支吸收余数

        # 多尺度频带提取
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch_low, kernel_size=5, stride=(2,1), padding=2),
                nn.BatchNorm2d(ch_low), # 归一化
                nn.GELU()
            ),  # 低频
            nn.Sequential(
                nn.Conv2d(ch, ch_mid, kernel_size=3,stride=(1,1), padding=1),
                nn.BatchNorm2d(ch_mid),  # 归一化
                nn.GELU()
            ),  # 中频
            nn.Sequential(
                nn.Conv2d(ch, ch_high, kernel_size=3, stride=(1,1), dilation=2, padding=2),
                nn.BatchNorm2d(ch_high),  # 归一化
                nn.GELU()
            )  # 高频
        ])
        # 归一化，避免 torch.cat 之后的数值过大
        self.norm = nn.BatchNorm2d(ch)

        # 动态频带注意力
        self.att = ChannelSpatialAttention(ch)


    def forward(self, x):
        # 多尺度特征提取
        low = F.avg_pool2d(self.convs[0](x), kernel_size=(2,1))  # 低频：平均池化进一步下采样
        mid = self.convs[1](x)                                  # 中频：保持原始分辨率
        high = F.max_pool2d(self.convs[2](x), kernel_size=(2,1)) # 高频：最大池化保留显著特征

        # 上采样对齐维度
        low = F.interpolate(low, size=x.shape[-2:], mode='nearest')   # 低频上采样
        high = F.interpolate(high, size=x.shape[-2:], mode='nearest') # 高频上采样

        # 拼接多尺度特征
        fused = torch.cat([low, mid, high], dim=1)  # [B, ch, H, W]
        fused = self.norm(fused)

        # 注意力加权
        out = fused * torch.sigmoid(self.att(fused))

        # 残差连接
        out = out + x
        return out


#谐波增强单元
class HarmonicEnhanceUnit(nn.Module):
    """谐波增强单元"""

    def __init__(self, ch):
        super().__init__()

        # 动态分配通道数
        base_ch = ch // 3  # 基准通道数
        remainder = ch % 3  # 余数
        ch_low = base_ch
        ch_mid = base_ch
        ch_high = base_ch + remainder  # 高频分支吸收余数

        # 多尺度谐波检测
        self.dilated_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch_low, kernel_size=3, dilation=1, padding=1),
                nn.BatchNorm2d(ch_low),  # 归一化
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(ch, ch_mid, kernel_size=3,  dilation=3, padding=3),
                nn.BatchNorm2d(ch_mid),  # 归一化
                nn.GELU()
            ),
            nn.Sequential(
                nn.Conv2d(ch, ch_high, kernel_size=3,  dilation=5, padding=5),
                nn.BatchNorm2d(ch_high),  # 归一化
                nn.GELU()
            )
        ])

        # 动态置信度检测
        self.confidence = nn.Sequential(
            nn.Conv2d(ch, ch // 8, 1),
            nn.GELU(),
            nn.Conv2d(ch // 8, len(self.dilated_convs), 1),  # 输出 3 个权重，对应 3 个分支
            nn.Softmax(dim=1)
        )

        self.conv = nn.Conv2d(ch, ch, 1)

        # 噪声门控（使用 Sigmoid）
        self.noise_gate = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # 提取谐波特征
        harmonics = [conv(x) for conv in self.dilated_convs]  # 列表，3 个 [B, ch//4, H, W]

        # 动态置信度检测
        weights = self.confidence(x)  # [B, 3, H, W]

        # 逐分支加权
        weighted_harmonics = [
            harmonics[0] * weights[:, 0:1, :, :],  # 高频: [B, 21, H, W]
            harmonics[1] * weights[:, 1:2, :, :],  # 中频: [B, 22, H, W]
            harmonics[2] * weights[:, 2:3, :, :]  # 低频: [B, 21, H, W]
        ]

        # 拼接并融合
        weighted = torch.cat(weighted_harmonics, dim=1)  # [B, 21+22+21=64, H, W]
        weighted = self.conv(weighted)  # [B, ch, H, W]

        # 噪声门控
        gate = self.noise_gate(x)  # [B, ch, H, W]
        out = gate * x + (1 - gate) * weighted

        # 残差连接
        out = self.dropout(out) + x
        return out


#时频全局融合模块
class TemporalBandGate(nn.Module):
    """时频联合动态融合模块"""

    def __init__(self, ch, reduction_ratio=4):
        super().__init__()
        # 1. 多尺度时频上下文提取（保护编码器细节）
        self.context = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch // reduction_ratio, kernel_size=k, padding=k // 2),
                nn.BatchNorm2d(ch // reduction_ratio),
                nn.GELU()
            ) for k in [3, 5]  # 3x3、5x5捕捉局部模式
        ])
        self.context_fuse = nn.Sequential(
            nn.Conv2d(ch // reduction_ratio * 2, ch, 1),  # 融合多尺度特征
            nn.BatchNorm2d(ch),
            nn.GELU()
        )

        # 2. 时序细节保护（1x5卷积增强局部动态）
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(ch, ch // reduction_ratio, kernel_size=(1, 5), padding=(0, 2)),  # 专注时序
            nn.BatchNorm2d(ch // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(ch // reduction_ratio, ch, 1),
            nn.Sigmoid()
        )

        # 3. 动态融合权重（平衡编码器与瓶颈层特征）
        self.gate = nn.Sequential(
            nn.Conv2d(ch, ch // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(ch // reduction_ratio, ch, 1),
            nn.Sigmoid()
        )

        # 4. 特征精炼（保护全局与局部细节）
        self.refine = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),  # 深度可分离卷积
            nn.BatchNorm2d(ch),
            nn.GELU(),
        )

        # 5. 残差路径
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)  # [B, C, F, T]

        # 1. 多尺度时频上下文提取
        ctx_3x3 = self.context[0](x)  # [B, C//4, F, T]
        ctx_5x5 = self.context[1](x)  # [B, C//4, F, T]
        ctx_fused = self.context_fuse(torch.cat([ctx_3x3, ctx_5x5], dim=1))  # [B, C, F, T]

        # 2. 时序细节保护
        temporal_weights = self.temporal_conv(x)  # [B, C, F, T]
        temporal_enhanced = ctx_fused * temporal_weights  # [B, C, F, T]

        # 3. 动态加权融合
        gate_weights = self.gate(temporal_enhanced)  # [B, C, F, T]
        gated_feat = temporal_enhanced * gate_weights  # [B, C, F, T]

        # 4. 特征精炼
        refined = self.refine(gated_feat)  # [B, C, F, T]

        # 5. 残差连接
        out = refined + residual
        return out

class BioFusionBlock(nn.Module):

    def __init__(self, out_ch,reduction_ratio=4):
        super().__init__()
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch // reduction_ratio, 3, padding=1),
            nn.BatchNorm2d(out_ch // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(out_ch // reduction_ratio, 2, 1),
            nn.Softmax(dim=1)  # 归一化权重
        )

        # 频带金字塔特征融合
        self.freq_pyramid = FrequencyPyramidUnit(out_ch)
        # 谐波增强单元
        self.harmonic = HarmonicEnhanceUnit(out_ch)
        #时频全局模块
        self.temporal = TemporalBandGate(out_ch)
        # 特征精炼
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),  # 深度卷积
            nn.GELU(),
            nn.BatchNorm2d(out_ch),
        )

        self.fusion_weights = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch // 8, 1),
            nn.GELU(),
            nn.Conv2d(out_ch // 8, 3, 1),
            nn.Softmax(dim=1)  # 为三个特征生成归一化权重
        )
    def forward(self, x, skip):
        # 动态融合
        concat_feat = torch.cat([x, skip], dim=1)
        gate_weights = self.fusion_gate(concat_feat)
        fused = x * gate_weights[:, 0:1] + skip * gate_weights[:, 1:2]
        # 多尺度处理
        pyramid_feat = self.freq_pyramid(fused)
        # 谐波结构增强
        harmonic_feat = self.harmonic(pyramid_feat)
        # 时序连续性建模
        temporal_feat = self.temporal(harmonic_feat)
        feat_stack = torch.cat([pyramid_feat, harmonic_feat, temporal_feat], dim=1)  # [B, 3*out_ch, F, T]
        weights = self.fusion_weights(feat_stack)  # [B, 3, F, T]
        fused_feat = (
                pyramid_feat * weights[:, 0:1] +
                harmonic_feat * weights[:, 1:2] +
                temporal_feat * weights[:, 2:3]
        )
        fused_feat = fused_feat + fused
        return self.refine(fused_feat)


#######################
# U-Net主干网络
#######################
#解码器
class BioDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),  # 调整通道数
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1),
            nn.BatchNorm2d(out_ch)
        )
        self.fusion = BioFusionBlock(out_ch)  # 使用改进的融合模块

    def forward(self, x, skip):
        x = self.up(x)
        return self.fusion(x, skip)

class UNetSpectroFusion(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        # 编码器
        self.encoder_block = nn.ModuleList()
        # 第一块编码器
        self.encoder_block.append(nn.Sequential(
            BirdFeature(1,32,use_channel_boost=False),
            nn.AvgPool2d(kernel_size=(2, 1))  # 仅在频率轴下采样
            )
        )
            #第二块编码器
        self.encoder_block.append(nn.Sequential(
            BirdFeature(32, 64,use_channel_boost=False),
            nn.AvgPool2d(kernel_size=(2, 1))  # 仅在频率轴下采样
            )
        )
            #第三块编码器
        self.encoder_block.append(nn.Sequential(
            BirdFeature(64, 128,use_channel_boost=False),
            nn.AvgPool2d(kernel_size=(2, 1))  # 在频率轴下采样
            )
        )
            #第四块编码器
        self.encoder_block.append(nn.Sequential(
            BirdFeature(128, 256,use_channel_boost=True),
         )
        )

        self.fusions = nn.Sequential(
            *[SpectroFusionBlock(dim=256) for _ in range(2)]
        )

        # 解码器（进行底层语义与高层语义的融合）
        self.decoder = nn.ModuleList([
            BioDecoder(256, 128,scale_factor=(1,1)),
            BioDecoder(128, 64,scale_factor=(2,1)),
            BioDecoder(64, 32,scale_factor=(2,1)),
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, None)),
            Rearrange('b c 1 t -> b (c t)'),
            nn.LayerNorm(32 * 32),
            nn.Linear(32 * 32, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

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
    model = UNetSpectroFusion(num_classes=20)
    start_time = time.time()  # 记录开始时间
    x = torch.randn(128, 1, 128, 32)  # Mel频谱图示例
    output = model(x)
    print(output.shape)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算运行时间
    print(f"代码运行耗时：{elapsed_time:.6f} 秒")