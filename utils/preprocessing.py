import cv2
import numpy as np
import torch
from torchvision import transforms

class OraclePreprocessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
    def preprocess(self, image):
        """改进的预处理方法，更好地保留小笔画"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. 使用OTSU阈值
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 2. 轻微的中值滤波去除噪点
        denoised = cv2.medianBlur(binary, 3)
        
        # 3. 连通区域分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(denoised, connectivity=8)
        
        # 4. 分析区域特征
        result = np.zeros_like(binary)
        
        # 找到主要连通区域
        areas = stats[1:, cv2.CC_STAT_AREA]  # 跳过背景
        if len(areas) > 0:
            max_area = np.max(areas)
            mean_area = np.mean(areas)
            min_area_threshold = mean_area * 0.05  # 降低面积阈值，更容易保留小笔画
            
            for i in range(1, num_labels):
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                            stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]
                
                # 计算长宽比
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                
                # 更宽松的笔画判断条件
                is_stroke = False
                
                # 1. 主要笔画判断
                if area > max_area * 0.1:
                    is_stroke = True
                # 2. 小笔画判断（更宽松的条件）
                elif area > min_area_threshold:
                    if aspect_ratio > 1.5:  # 降低长宽比要求
                        is_stroke = True
                    elif w >= 3 or h >= 3:  # 降低最小尺寸要求
                        is_stroke = True
                # 3. 特别小的笔画（点、短横等）
                elif w >= 2 and h >= 2:  # 保留非常小的但确实是笔画的部分
                    is_stroke = True
                
                if is_stroke:
                    result[labels == i] = 255
        
        return result
    
    def get_binary_target(self, image):
        """目标图像需要完整的预处理"""
        # 使用完整的预处理方法生成目标
        processed = self.preprocess(image)
        
        # 调整大小
        resized = cv2.resize(processed, self.target_size, 
                           interpolation=cv2.INTER_NEAREST)
        
        # 转换为[0,1]范围的tensor
        tensor = torch.FloatTensor(resized / 255.0).unsqueeze(0)
        
        return tensor
    
    def __call__(self, image):
        """输入图像只需要基础预处理"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 只调整大小，保持原始灰度信息
        resized = cv2.resize(image, self.target_size, 
                           interpolation=cv2.INTER_AREA)
        
        # 转换为tensor并归一化
        tensor = self.transform(resized)
        
        return tensor 