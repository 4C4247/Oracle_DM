import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DirectionalConv(nn.Module):
    """保留原有的方向性卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 保持原有的水平和垂直卷积结构
        self.h_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels//4, kernel_size=(1, k), padding=(0, k//2))
            for k in [3, 5, 7]
        ])
        self.v_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels//4, kernel_size=(k, 1), padding=(k//2, 0))
            for k in [3, 5, 7]
        ])
        
        # 原有的特征融合和注意力模块
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 3//2, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dir_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 原有的方向特征提取
        h_feats = [conv(x) for conv in self.h_convs]
        v_feats = [conv(x) for conv in self.v_convs]
        
        h_out = torch.cat(h_feats, dim=1)
        v_out = torch.cat(v_feats, dim=1)
        
        combined = torch.cat([h_out, v_out], dim=1)
        fused = self.fusion(combined)
        
        # 使用注意力机制
        attention = self.dir_attention(fused)
        out = fused * attention
        
        return out  # 只返回特征图

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dir_conv = DirectionalConv(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.dir_conv(x)
        return self.conv(x)

class SSIM(nn.Module):
    """结构相似性损失"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer('window', self.create_window(window_size))
        
    def create_window(self, window_size):
        # 创建高斯窗口
        sigma = 1.5
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) 
                            for x in range(window_size)])
        gauss = gauss/gauss.sum()
        
        # 创建2D窗口
        _2D_window = gauss.unsqueeze(1) * gauss.unsqueeze(0)
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        return window
        
    def forward(self, img1, img2):
        # 确保输入维度正确
        if len(img1.shape) != 4:
            img1 = img1.unsqueeze(0)
        if len(img2.shape) != 4:
            img2 = img2.unsqueeze(0)
        
        # 数值稳定性
        img1 = torch.clamp(img1, min=1e-6, max=1-1e-6)
        img2 = torch.clamp(img2, min=1e-6, max=1-1e-6)
        
        window = self.window.expand(img1.shape[1], 1, self.window_size, self.window_size)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=img2.shape[1])
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=img2.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=img1.shape[1]) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # 返回1-SSIM作为损失
        return 1 - ssim_map.mean()

class StrokeExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        
        # 使用更少的通道数
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 从64减到32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 从128减到64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 从256减到128
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        
        # 简化注意力模块
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 简化解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 特征提取
        features = self.encoder(x)
        
        # 注意力加权
        attention = self.attention(features)
        attended_features = features * attention
        
        # 解码生成笔画
        output = self.decoder(attended_features)
        
        return output

    def extract_features(self, x):
        """提取中间层特征用于后续TDM"""
        features = self.encoder(x)
        return features

class StrokeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()
        
        # 预计算并注册所有卷积核
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('noise_kernel', torch.ones(5, 5).view(1, 1, 5, 5))
        
        # 保持原有的SSIM窗口大小
        self.ssim = SSIM(window_size=11)
        
        # 初始化损失值字典
        self.loss_values = {
            'bce': 0.0,
            'edge': 0.0,
            'ssim': 0.0,
            'continuity': 0.0,
            'noise': 0.0
        }
    
    def compute_noise_suppression(self, pred, target):
        """计算噪点抑制损失"""
        # 确保kernel维度正确
        local_sum = F.conv2d(pred, self.noise_kernel, padding=2)  # padding应该是2而不是1，因为是5x5的kernel
        noise_mask = (local_sum < 3) * pred  # 阈值调整为3
        return torch.mean(noise_mask)

    def forward(self, pred, target):
        # 确保所有输入都有正确的维度
        if len(pred.shape) != 4:
            pred = pred.unsqueeze(0)
        if len(target.shape) != 4:
            target = target.unsqueeze(0)
        
        # 1. 基础重建损失
        bce_loss = self.bce(pred, target)
        
        # 2. 边缘一致性损失
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
        
        edge_loss = F.mse_loss(pred_edges_x, target_edges_x) + F.mse_loss(pred_edges_y, target_edges_y)
        
        # 3. 结构相似性损失
        ssim_loss = self.ssim(pred, target)
        
        # 4. 笔画连续性损失
        continuity_loss = self.compute_continuity_loss(pred)
        
        # 5. 噪点抑制损失
        noise_loss = self.compute_noise_suppression(pred, target)
        
        # 计算总损失
        total_loss = (bce_loss * 1.0 + 
                     edge_loss * 0.3 +
                     ssim_loss * 0.2 +
                     continuity_loss * 0.2 +
                     noise_loss * 0.3)
        
        # 更新损失值字典
        self.loss_values.update({
            'bce': bce_loss.item(),
            'edge': edge_loss.item(),
            'ssim': ssim_loss.item(),
            'continuity': continuity_loss.item(),
            'noise': noise_loss.item()
        })
        
        return total_loss
        
    def compute_continuity_loss(self, pred):
        """计算笔画连续性损失"""
        # 水平和垂直方向的梯度
        dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        
        # 鼓励笔画的连续性
        continuity = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        
        # 惩罚突变
        smoothness = torch.mean(torch.abs(dx[:, :, :, 1:] - dx[:, :, :, :-1])) + \
                    torch.mean(torch.abs(dy[:, :, 1:, :] - dy[:, :, :-1, :]))
                    
        return continuity + 0.5 * smoothness
        
    def compute_noise_suppression(self, pred, target):
        """计算噪点抑制损失"""
        # 计算局部区域的连通性
        kernel = torch.ones(5, 5).to(pred.device)
        local_sum = F.conv2d(pred, kernel.view(1, 1, 5, 5), padding=2)
        
        # 孤立点（噪点）应该有较小的局部和
        noise_mask = (local_sum < 3) * pred
        
        # 抑制噪点
        noise_loss = torch.mean(noise_mask)
        
        return noise_loss 