import torch
import cv2
import os
import numpy as np
from utils.preprocessing import OraclePreprocessor

def extract_and_save_strokes(image_path, stroke_extractor, save_dir):
    """提取笔画并保存，用于超分辨率模型"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # 预处理
    preprocessor = OraclePreprocessor(target_size=(256, 256))
    processed_tensor = preprocessor(image).unsqueeze(0)
    
    # 提取笔画
    with torch.no_grad():
        strokes = stroke_extractor(processed_tensor)
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.basename(image_path)
        # 保存笔画图
        stroke_output = (strokes.cpu().numpy()[0, 0] * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"{base_name}_strokes.png"), stroke_output)
    
    return strokes 