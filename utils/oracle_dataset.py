import os
import torch
from torch.utils.data import Dataset
import cv2
from utils.preprocessing import OraclePreprocessor

class OracleDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        # 过滤掉无法读取的图像
        self.image_files = []
        for f in os.listdir(data_dir):
            if f.endswith('.jpg'):
                img_path = os.path.join(data_dir, f)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        self.image_files.append(f)
                    else:
                        print(f"Warning: Could not read image {f}")
                except Exception as e:
                    print(f"Error reading {f}: {str(e)}")
        
        # 使用与测试相同的预处理器
        self.preprocessor = OraclePreprocessor(target_size=(256, 256))
        print(f"Successfully loaded {len(self.image_files)} images")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # 读取原始图像
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = cv2.imread(img_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
            
        try:
            # 使用与测试相同的预处理方式
            processed = self.preprocessor(image)
            target = self.preprocessor.get_binary_target(image)
            
            return processed, target
            
        except Exception as e:
            print(f"Error processing image {img_path}: {str(e)}")
            # 返回一个空白图像作为替代
            blank = torch.zeros((1, 256, 256), dtype=torch.float32)
            return blank, blank 