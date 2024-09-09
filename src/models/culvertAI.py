import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os


# Depthwise Separable Convolution Layer
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Attention Router for Dynamic Routing
class AttentionRouter(nn.Module):
    def __init__(self, in_channels, num_routes):
        super(AttentionRouter, self).__init__()
        self.attention = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            DepthwiseSeparableConv(64, num_routes, kernel_size=1),
        )

    def forward(self, x):
        weights = self.attention(x)
        return F.softmax(weights, dim=1)

# Dynamic FPN Block with Routing
class DynamicFPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_routes=3):
        super(DynamicFPNBlock, self).__init__()
        self.router = AttentionRouter(in_channels, num_routes)
        self.routes = nn.ModuleList([
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1),
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=5, padding=2),
            DepthwiseSeparableConv(in_channels, out_channels, kernel_size=7, padding=3)
        ])

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        weights = self.router(x)
        results = [route(x) for route in self.routes]

        weights = weights.unsqueeze(2)

        weighted_results = []
        for i in range(len(results)):
            weighted_results.append(weights[:, i] * results[i])
        output = sum(weighted_results)

        return F.relu(self.norm(output))

# Squeeze-and-Excitation Block
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = max(1, channels // reduction_ratio)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y

# Residual MBConv Block
class ResidualMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1, use_se=True):
        super(ResidualMBConv, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * expansion_factor

        self.expand = DepthwiseSeparableConv(in_channels, hidden_dim, 1) if expansion_factor != 1 else nn.Identity()
        self.dwconv = DepthwiseSeparableConv(hidden_dim, hidden_dim, 3, stride, 1)
        self.se = SqueezeExcitation(hidden_dim) if use_se else nn.Identity()
        self.project = DepthwiseSeparableConv(hidden_dim, out_channels, 1)

        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.bn1(F.silu(x))
        x = self.bn2(F.silu(self.dwconv(x)))
        x = self.se(x)
        x = self.bn3(self.project(x))
        if self.use_residual:
            x += residual
        return x

# Dilated Convolution Block
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConvBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=2, stride=2)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=4, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.bn(x)



# Dynamic FPN with Attention Routing (Main Model)
class DynamicFPNWithAttentionRouting(nn.Module):
    def __init__(self, num_classes, width_mult=0.75):
        super(DynamicFPNWithAttentionRouting, self).__init__()

        def scaled_width(width):
            return max(8, int(width * width_mult))  # Ensure minimum width of 8

        self.stem = nn.Sequential(
            DepthwiseSeparableConv(3, scaled_width(32), 3, 2, 1),
            nn.BatchNorm2d(scaled_width(32)),
            nn.SiLU()
        )

        self.block1 = ResidualMBConv(scaled_width(32), scaled_width(16), expansion_factor=1)
        self.block2 = ResidualMBConv(scaled_width(16), scaled_width(32), stride=2)
        self.block3 = DilatedConvBlock(scaled_width(32), scaled_width(64))

        fpn_channels = scaled_width(256)
        self.fpn_blocks = nn.ModuleList([
            DynamicFPNBlock(scaled_width(64), fpn_channels),
            DynamicFPNBlock(scaled_width(64), fpn_channels),  # Newly added block
            DynamicFPNBlock(scaled_width(32), fpn_channels),
            DynamicFPNBlock(scaled_width(16), fpn_channels)
        ])

        # Add projection layers for proper channel matching
        self.proj_p4_p3 = DepthwiseSeparableConv(fpn_channels, scaled_width(64), kernel_size=1)
        self.proj_p3_p2 = DepthwiseSeparableConv(fpn_channels, scaled_width(32), kernel_size=1)
        self.proj_p2_p1 = DepthwiseSeparableConv(fpn_channels, scaled_width(16), kernel_size=1)

        self.pred_heads = nn.ModuleList([
            DepthwiseSeparableConv(fpn_channels, num_classes, kernel_size=3, padding=1) for _ in range(4)
        ])

        self.fusion = DepthwiseSeparableConv(num_classes * 4, num_classes, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)

        p4 = self.fpn_blocks[0](f3)

        p4_upsampled = F.interpolate(p4, size=f3.shape[2:], mode='nearest')
        p4_projected = self.proj_p4_p3(p4_upsampled)
        p3 = self.fpn_blocks[1](f3 + p4_projected)

        p3_upsampled = F.interpolate(p3, size=f2.shape[2:], mode='nearest')
        p3_projected = self.proj_p3_p2(p3_upsampled)
        p2 = self.fpn_blocks[2](f2 + p3_projected)

        p2_upsampled = F.interpolate(p2, size=f1.shape[2:], mode='nearest')
        p2_projected = self.proj_p2_p1(p2_upsampled)
        p1 = self.fpn_blocks[3](f1 + p2_projected)

        preds = [head(p) for head, p in zip(self.pred_heads, [p1, p2, p3, p4])]

        fused = self.fusion(torch.cat([F.interpolate(p, size=x.shape[2:], mode='bilinear', align_corners=False) for p in preds], dim=1))
        fused = self.upsample(fused)

        if fused.shape[2:] != x.shape[2:]:
            fused = F.interpolate(fused, size=x.shape[2:], mode='bilinear', align_corners=False)

        return fused

def attempt_load(weights, width_mult, device):
    num_classes = 9
    model = DynamicFPNWithAttentionRouting(num_classes=num_classes, width_mult=width_mult).to(device)
    model.load_state_dict(torch.load(weights))
    return model
