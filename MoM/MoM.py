import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class MoM(nn.Module):
    """Mixture of Modifiers"""
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_blocks=8):
        super().__init__()
        
        # 初始特征提取 - 修改输入通道数
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResBlock(hidden_channels))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 特征融合 - 确保输出通道数正确
        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1)  # 输出通道数与输入相同
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 初始特征提取
        feat = self.conv_in(x)
        
        # 残差特征提取
        res_feat = self.res_blocks(feat)
        
        # 注意力权重
        att = self.attention(res_feat)
        
        # 特征融合
        out = self.conv_out(res_feat * att)
        
        return out
        

