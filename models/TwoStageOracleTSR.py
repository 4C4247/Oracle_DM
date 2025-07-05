import torch
import torch.nn as nn
from model.OracleDenoisingModel import OracleDenoisingModel
from model.OracleDiffTSR import OracleDiffTSR
from model.IDM.modules.diffusionmodules.util import extract_into_tensor
import logging
from models.stroke_extractor import StrokeExtractor
import torch.nn.functional as F
import math

logger = logging.getLogger(__name__)

class TwoStageOracleTSR(nn.Module):
    def __init__(self, sr_config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化超分模型
        self.sr_model = OracleDiffTSR(**sr_config)
        
    def super_resolve(self, x, strokes, scale_factor=2):
        """优化的超分辨率处理，保持笔画引导"""
        if self.training:
            # 减少时间步但保持笔画引导
            t = torch.randint(0, self.sr_model.timesteps // 4, (x.size(0),), device=x.device)
            
            # 计算目标尺寸（确保是8的倍数）
            orig_h, orig_w = x.shape[2:]
            target_h = ((orig_h * scale_factor + 7) // 8) * 8
            target_w = ((orig_w * scale_factor + 7) // 8) * 8
            
            # 使用更高效的上采样，但保持笔画信息
            x_upsampled = F.interpolate(x, size=(target_h, target_w), 
                                      mode='bilinear', align_corners=False)
            strokes_upsampled = F.interpolate(strokes, size=(target_h, target_w), 
                                            mode='nearest')  # 使用nearest保持笔画边缘清晰
            
            # 添加自适应噪声，基于笔画
            noise_scale = 0.01 * strokes_upsampled  # 减小噪声但保持笔画引导
            noise = torch.randn_like(x_upsampled) * noise_scale
            noised_x = x_upsampled + noise
            
            # 确保输入和模型参数类型一致
            enhanced = self.sr_model(noised_x.float(), strokes_upsampled.float(), t)
            
            return enhanced
        else:
            t = torch.full((x.size(0),), self.sr_model.timesteps - 1, 
                          dtype=torch.long, device=x.device)
            
            # 计算目标尺寸（确保是8的倍数）
            orig_h, orig_w = x.shape[2:]
            target_h = ((orig_h * scale_factor + 7) // 8) * 8
            target_w = ((orig_w * scale_factor + 7) // 8) * 8
            
            x_upsampled = F.interpolate(x, size=(target_h, target_w), 
                                      mode='bicubic', align_corners=False)
            strokes_upsampled = F.interpolate(strokes, size=(target_h, target_w), 
                                            mode='bicubic', align_corners=False)
            
            noised_x = x_upsampled + torch.randn_like(x_upsampled) * 0.005 * strokes_upsampled
        
        # 生成增强结果
        enhanced = self.sr_model(noised_x, strokes_upsampled, t)
        
        return enhanced  # 不再进行额外的边缘增强和平滑

    def enhance_edges_and_smooth(self, x, strokes):
        """增强边缘并平滑处理"""
        # Sobel边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).float()
        
        grad_x = F.conv2d(x, sobel_x.view(1,1,3,3), padding=1)
        grad_y = F.conv2d(x, sobel_y.view(1,1,3,3), padding=1)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 自适应边缘增强，只在笔画区域
        edge_weight = 0.1 * strokes  # 笔画区域加强边缘
        enhanced = x + edge_weight * grad_mag * torch.sign(x)
        
        # 高斯平滑处理边缘
        kernel_size = 3
        sigma = 0.5
        gaussian_kernel = self.get_gaussian_kernel(kernel_size, sigma)
        smoothed = F.conv2d(enhanced, gaussian_kernel, padding=kernel_size//2)
        
        # 只在笔画边缘区域应用平滑
        edge_mask = (grad_mag > 0.1).float() * strokes
        result = enhanced * (1 - edge_mask) + smoothed * edge_mask
        
        return result

    def get_gaussian_kernel(self, kernel_size=3, sigma=0.5):
        """生成高斯核"""
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
        
        mean = (kernel_size - 1)/2.
        variance = sigma**2
        gaussian_kernel = (1./(2.*math.pi*variance)) * \
                         torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance))
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        
        return gaussian_kernel.view(1, 1, kernel_size, kernel_size).to(self.device)

    def enhance_edges(self, x):
        """边缘增强处理"""
        # Sobel算子检测边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device).float()
        
        # 计算梯度
        grad_x = F.conv2d(x, sobel_x.view(1,1,3,3), padding=1)
        grad_y = F.conv2d(x, sobel_y.view(1,1,3,3), padding=1)
        
        # 计算梯度幅值
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        
        # 降低边缘增强的强度
        enhanced = x + 0.1 * grad_mag * torch.sign(x)  # 从0.2降低到0.1
        
        return enhanced

    def enhance_edges_multiscale(self, x):
        """多尺度边缘增强"""
        scales = [1, 0.75]  # 减少尺度数量，避免过度处理
        enhanced = x
        
        for scale in scales:
            size = (int(x.shape[2] * scale), int(x.shape[3] * scale))
            if scale != 1:
                curr_x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
            else:
                curr_x = x
            
            # 边缘增强
            edge_enhanced = self.enhance_edges(curr_x)
            
            if scale != 1:
                edge_enhanced = F.interpolate(edge_enhanced, 
                                           size=(x.shape[2], x.shape[3]), 
                                           mode='bilinear', 
                                           align_corners=False)
            
            enhanced = enhanced + 0.1 * edge_enhanced  # 降低增强强度
        
        return enhanced / (1 + 0.1 * len(scales))

    def inference(self, x, strokes, scale_factor=2):
        """推理时的超分辨率处理"""
        # 先生成增强结果
        enhanced = self.super_resolve(x, strokes, scale_factor)
        
        # 然后进行放大
        target_size = (x.shape[2] * scale_factor, x.shape[3] * scale_factor)
        sr_output = F.interpolate(
            enhanced,
            size=target_size,
            mode='bicubic',  # 使用双三次插值
            align_corners=False
        )
        
        # 最后再次进行边缘增强
        sr_output = self.enhance_edges(sr_output)
        
        return sr_output
    
    def forward(self, x, strokes):
        return self.super_resolve(x, strokes)

    def compute_sharpness_loss(self, pred):
        """改进的锐利度损失计算"""
        # 添加数值稳定性检查
        if torch.isnan(pred).any():
            return torch.tensor(0.0, device=pred.device)
        
        # Laplacian算子
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=pred.device, dtype=torch.float32)
        
        # 计算Laplacian响应
        laplacian = F.conv2d(
            pred,
            laplacian_kernel.view(1, 1, 3, 3),
            padding=1
        )
        
        # 计算梯度
        grad_x = F.conv2d(
            pred, 
            torch.tensor([[-1., 0., 1.]], device=pred.device).view(1, 1, 1, 3),
            padding=(0, 1)
        )
        grad_y = F.conv2d(
            pred, 
            torch.tensor([[-1.], [0.], [1.]], device=pred.device).view(1, 1, 3, 1),
            padding=(1, 0)
        )
        
        # 计算梯度幅值，添加数值稳定性
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        # 使用更稳定的损失计算
        sharpness_loss = torch.mean(torch.abs(laplacian)) + torch.mean(gradient_mag)
        
        # 添加值范围检查
        if torch.isnan(sharpness_loss) or torch.isinf(sharpness_loss):
            return torch.tensor(0.0, device=pred.device)
        
        return sharpness_loss 