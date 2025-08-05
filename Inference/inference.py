import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import cv2
import numpy as np
from models.TwoStageOracleTSR import TwoStageOracleTSR
from models.stroke_extractor import StrokeExtractor
from omegaconf import OmegaConf
import os
from pathlib import Path
import time

# def process_single_image(sr_model, stroke_extractor, image_path, output_dir, target_size=(256, 256)):
#     print(f"处理图像: {image_path}")
#     inference_with_stroke(sr_model, stroke_extractor, image_path, output_dir, target_size, 
#                          )  
   

def process_batch_images(sr_model, stroke_extractor, input_dir, output_dir, target_size=(256, 256)):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    for ext in img_extensions:
        image_files.extend(list(input_path.glob(f'*{ext}')))
        image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
    total_files = len(image_files)
    if total_files == 0:
        print(f"without image files in {input_dir}")
        return
    print(f"found {total_files} image files, start batch processing...")
    start_time = time.time()
    for i, img_path in enumerate(image_files):
        print(f"[{i+1}/{total_files}] processing: {img_path.name}")
        try:
            inference_with_stroke(sr_model, stroke_extractor, str(img_path), output_dir, target_size,
                                 # stroke_dir=stroke_dir, sr_dir=sr_dir
                                 )
        except Exception as e:
            print(f"error processing {img_path.name}: {e}")
    elapsed_time = time.time() - start_time
    print(f"batch processing completed! processed {total_files} files, time taken: {elapsed_time:.2f} seconds")
    #   print(f"笔画提取结果保存至: {stroke_dir}")
    # print(f"超分辨率结果保存至: {sr_dir}")

if __name__ == '__main__':
    PROCESS_MODE = "single"
    MODEL_CONFIG_PATH = "configs/two_stage_oracle_config.yaml"
    MODEL_CHECKPOINT_PATH = "outputs/checkpoints/best_model.pth"
    INPUT_PATH = "K:\\foxdownload\\result2\\val_32\pic908.jpg"
    OUTPUT_DIR = "E:/DiffTSR-main/outputs/samples/paper_inference_edge_pixelshuffle"
    TARGET_SIZE = (256, 256)
    print("=== Edge-based PixelShuffle TSR ===")
    print(f"processing mode: {'single image' if PROCESS_MODE == 'single' else 'batch processing'}")
    print(f"config file: {MODEL_CONFIG_PATH}")
    print(f"model checkpoint: {MODEL_CHECKPOINT_PATH}")
    print(f"{'input image' if PROCESS_MODE == 'single' else 'input directory'}: {INPUT_PATH}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"目标尺寸: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    print("========================")
    sr_model = load_models(MODEL_CONFIG_PATH, MODEL_CHECKPOINT_PATH)
    stroke_extractor = sr_model.sr_model.stroke_extractor
    if PROCESS_MODE == "single":
        process_single_image(sr_model, stroke_extractor, INPUT_PATH, OUTPUT_DIR, TARGET_SIZE)
    else:
        process_batch_images(sr_model, stroke_extractor, INPUT_PATH, OUTPUT_DIR, TARGET_SIZE) 