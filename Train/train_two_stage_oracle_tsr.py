

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import cv2
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import wandb
from omegaconf import OmegaConf
import torch.nn.functional as F
from models.stroke_extractor import StrokeExtractor
import gc
import traceback
from torch.cuda.amp import autocast, GradScaler
from torchvision.datasets import ImageFolder
from PIL import Image
import random
import glob

from model.TwoStageOracleTSR import TwoStageOracleTSR

# log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# at the beginning of the program, clean up the CUDA cache
torch.cuda.empty_cache()
gc.collect()

class LowResDataset(Dataset):
    def __init__(self, low_res_dir, transform=None, subset_ratio=1.0):
        """
        Args:
            low_res_dir: 低分辨率图像目录
            transform: 数据转换
            subset_ratio: 使用数据集的比例
        """
        self.low_res_dir = Path(low_res_dir)
        self.transform = transform
        
        # get all image files 
        self.low_res_files = list(self.low_res_dir.glob('*.jpg'))
        
        # if use subset
        if subset_ratio < 1.0:
            num_files = len(self.low_res_files)
            subset_size = int(num_files * subset_ratio)
            self.low_res_files = self.low_res_files[:subset_size]
        
        logger.info(f"Dataset initialized with low-res images:")
        logger.info(f"- Low-res images dir: {self.low_res_dir}")
        logger.info(f"- Total low-res images: {len(self.low_res_files)}")

    def __len__(self):
        return len(self.low_res_files)

    def __getitem__(self, idx):
        """get a single sample"""
        # load low-resolution image
        lr_path = self.low_res_files[idx]
        lr_img = Image.open(lr_path).convert('L')

        hr_path = lr_path
        hr_img = Image.open(hr_path).convert('L')
        
        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)
        
        return lr_img, hr_img

class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_subset=True, subset_ratio=0.2):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        self.image_files = list(self.data_dir.glob('*.jpg')) + \
                          list(self.data_dir.glob('*.png')) + \
                          list(self.data_dir.glob('*.jpeg'))
        
        if use_subset:
            num_total = len(self.image_files)
            num_subset = int(num_total * subset_ratio)
            rng = np.random.RandomState(42)
            subset_indices = rng.choice(num_total, num_subset, replace=False)
            self.image_files = [self.image_files[i] for i in subset_indices]
            print(f"Using {len(self.image_files)} images (subset_ratio={subset_ratio}) from {num_total} images in {data_dir}")
        else:
            print(f"Using all {len(self.image_files)} images in {data_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            
        return image, image  

def verify_dataset(noisy_dir, clean_dir):
    """verify the completeness of the dataset"""
    noisy_dir = Path(noisy_dir)
    clean_dir = Path(clean_dir)
    
    # check if the directory exists
    if not noisy_dir.exists():
        logger.error(f"Noisy image directory not found: {noisy_dir}")
        return False
    if not clean_dir.exists():
        logger.error(f"Clean image directory not found: {clean_dir}")
        return False
    
    # get the file list
    noisy_files = set(f.name for f in noisy_dir.glob('*.jpg'))
    clean_files = set(f.name for f in clean_dir.glob('*.jpg'))
    
    # check the number of files
    logger.info(f"Found {len(noisy_files)} noisy images")
    logger.info(f"Found {len(clean_files)} clean images")
    

    min_count = min(len(noisy_files), len(clean_files))
    if min_count < 100: 
        logger.warning(f"Very few training pairs found: {min_count}")
    
    return min_count > 0

class TwoStageTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # set the save path
        self.save_dir = Path(config.training.save_dir)
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir = self.save_dir / 'samples'
        
        # create the necessary directories
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize the model
        self.model = TwoStageOracleTSR(config.sr_model).to(self.device)
        self.stroke_extractor = StrokeExtractor(in_channels=1).to(self.device)
        
        # initialize the start epoch
        self.start_epoch = 0
        
        # load the pretrained stroke extractor
        self.load_pretrained_stroke_extractor(config.stroke_extractor.pretrained_path)
        
        # initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # initialize the data loader
        self.train_loader = self.create_dataloader(
            config.data.clean_train_dir,
            batch_size=config.training.batch_size,
            is_train=True
        )
        self.val_loader = self.create_dataloader(
            config.data.clean_val_dir,
            batch_size=config.training.batch_size,
            is_train=False
        )
        
        # set the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=config.training.get('min_lr', 1e-7),  
            verbose=True
        )
        
        # if need to resume training
        if config.training.resume_training:
            self.resume_from_checkpoint()
        
        self.last_good_checkpoint = None  # add this line to track the last good checkpoint
        self.global_step = 0  # add the global step counter
        
        # add the Sobel operator for edge detection
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                   dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
        
        # add the mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

    def resume_from_checkpoint(self):
        """resume from the checkpoint"""
        checkpoint_path = self.config.training.resume_checkpoint
        
        # if use the latest checkpoint
        if self.config.training.resume_latest:
            checkpoint_dir = Path(self.config.training.save_dir) / 'checkpoints'
            checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            if checkpoints:
                checkpoint_path = str(sorted(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))[-1])
        
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # load the model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            
            print(f"Resumed from epoch {self.start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")

    def create_dataloader(self, data_dir, batch_size, is_train=True):
        """create the data loader, keep the original size"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        dataset = LowResDataset(
            low_res_dir=data_dir,
            transform=transform,
            subset_ratio=0.2
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )

    def validate(self):

        self.model.eval()
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        n_val_batches = 0
        
        try:
            with torch.no_grad():
                for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.val_loader):
                    lr_imgs = lr_imgs.to(self.device)
                    hr_imgs = hr_imgs.to(self.device)
                    
                    # get the stroke features
                    strokes = self.stroke_extractor(lr_imgs)
                    
                    # generate the super-resolution image
                    sr_output = self.model.super_resolve(lr_imgs, strokes, scale_factor=2)
                    
                    if torch.isnan(sr_output).any():
                        print(f"Warning: NaN detected in validation output at batch {batch_idx}")
                        continue
                    
                    batch_loss = (
                        mse_loss * self.config.training.loss_weights.mse +
                        edge_loss * self.config.training.loss_weights.edge +
                        (1 - ssim_val) * self.config.training.loss_weights.ssim
                    )
                    
                    val_loss += batch_loss.item()
                    val_psnr += batch_psnr
                    val_ssim += ssim_val
                    n_val_batches += 1
            
            if n_val_batches > 0:
                val_loss /= n_val_batches
                val_psnr /= n_val_batches
                val_ssim /= n_val_batches
            else:
                print("Warning: No valid validation batches!")
                val_loss = float('inf')
                val_psnr = 0
                val_ssim = 0
            
        except Exception as e:
            print(f"Error during validation: {e}")
            traceback.print_exc()
            val_loss = float('inf')
            val_psnr = 0
            val_ssim = 0
        
        return val_loss, val_psnr, val_ssim

    def compute_psnr(self, pred, target):
      
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        max_val = 1.0
        return 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse)

    def compute_ssim(self, pred, target, window_size=11):
      
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
      
        kernel = self.create_gaussian_kernel(window_size).to(pred.device)
        
   
        mu1 = F.conv2d(pred, kernel, padding=window_size//2)
        mu2 = F.conv2d(target, kernel, padding=window_size//2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
       
        sigma1_sq = F.conv2d(pred * pred, kernel, padding=window_size//2) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=window_size//2) - mu2_sq
        sigma12 = F.conv2d(pred * target, kernel, padding=window_size//2) - mu1_mu2
        
 
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.mean()

    def create_gaussian_kernel(self, window_size, sigma=1.5):
        """创建高斯核用于SSIM计算"""
        coords = torch.arange(window_size, dtype=torch.float32)
        coords -= window_size // 2
        
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        kernel = g.view(1, 1, -1) * g.view(1, -1, 1)
        return kernel.view(1, 1, window_size, window_size)

    def compute_sharpness_loss(self, pred):
 
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], device=pred.device, dtype=torch.float32).view(1, 1, 3, 3)
        
   
        laplacian = F.conv2d(pred, laplacian_kernel, padding=1)
        
        grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        # calculate the gradient magnitude
        gradient_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        return torch.mean(torch.abs(laplacian)) + torch.mean(gradient_mag)

    def edge_loss(self, pred, target):
        """calculate the edge loss"""
        # calculate the edge in x and y direction
        pred_edges_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_edges_y = F.conv2d(pred, self.sobel_y, padding=1)
        target_edges_x = F.conv2d(target, self.sobel_x, padding=1)
        target_edges_y = F.conv2d(target, self.sobel_y, padding=1)
        
        # calculate the edge loss
        edge_loss = F.mse_loss(pred_edges_x, target_edges_x) + \
                   F.mse_loss(pred_edges_y, target_edges_y)
        return edge_loss

    def train_step(self, lr_imgs, hr_imgs):
        """optimized training step, reduce the memory usage"""
        self.optimizer.zero_grad()
        
        # extract the stroke features
        with torch.no_grad(): 
            strokes = self.stroke_extractor(lr_imgs)
        
        
        try:
            # calculate the loss
            mse_loss = F.mse_loss(sr_output, target_hr)
            edge_loss = self.edge_loss(sr_output, target_hr)
            ssim_loss = 1 - self.compute_ssim(sr_output, target_hr)
            
            # reduce the loss items
            loss = (
                mse_loss * self.config.training.loss_weights.mse +
                edge_loss * self.config.training.loss_weights.edge +
                ssim_loss * self.config.training.loss_weights.ssim
            )
            
            # backward propagation
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
            
            self.optimizer.step()
            
            # add the sample save logic
            if self.global_step % self.config.training.sample_every == 0:
                sr_outputs = {scale: sr_output.detach()}
                self.save_samples(lr_imgs, sr_outputs, hr_imgs, self.global_step)
            
            self.global_step += 1  # ensure the global_step is updated
            return loss.item()
            
        except RuntimeError as e:
            print(f"Error in training step: {e}")
            return float('inf')

    def save_samples(self, lr_imgs, sr_outputs, hr_imgs, step):
        """save the samples in the training process"""
        sample_dir = self.sample_dir / f'step_{step}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # save the first image in a batch
        for i in range(min(4, lr_imgs.size(0))):
            # save the low-resolution input
            save_image(lr_imgs[i:i+1], sample_dir / f'lr_{i}.png')
            
            # save the super-resolution outputs in different scales
            for scale, sr_output in sr_outputs.items():
                save_image(sr_output[i:i+1], sample_dir / f'sr_x{scale}_{i}.png')
            
            # save the high-resolution target
            save_image(hr_imgs[i:i+1], sample_dir / f'hr_{i}.png')

    def load_pretrained_stroke_extractor(self, path):
        """load the pretrained stroke extractor"""
        checkpoint = torch.load(path)
        self.stroke_extractor.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self.stroke_extractor.eval()
    
    def train(self):
        """complete the training process"""
        best_val_loss = float('inf')
        
        # start from the saved epoch
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs}")
            
            self.model.train()
            total_loss = 0
            pbar = tqdm(self.train_loader)
            
            for batch_idx, (lr_imgs, hr_imgs) in enumerate(pbar):
                try:
   
                    lr_imgs = lr_imgs.to(self.device)
                    hr_imgs = hr_imgs.to(self.device)
                    
              
                    loss = self.train_step(lr_imgs, hr_imgs)
                    total_loss += loss
                    
                
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}'
                    })
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
            
           
            avg_train_loss = total_loss / len(self.train_loader)
            
          
            val_loss, val_psnr, val_ssim = self.validate()
            print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.4f}, Val SSIM: {val_ssim:.4f}")
            
           
            self.scheduler.step(val_loss)
            
            
            if epoch % self.config.training.save_every == 0:
                self.save_checkpoint(epoch, val_loss)
            
           
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """save the checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'global_step': self.global_step 
            }
            
            # use Path to handle the path
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            
            # save the current checkpoint
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # if this is a good checkpoint, record it
            if not isinstance(val_loss, float) or not np.isnan(val_loss):
                self.last_good_checkpoint = checkpoint_path
                
            # save the best model
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(checkpoint, best_path)
                print(f"Saved best model to {best_path}")
                
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")

    def load_last_good_checkpoint(self):
        """load the last good checkpoint"""
        if self.last_good_checkpoint is None:
            # if there is no good checkpoint, try to find the latest one
            checkpoint_dir = os.path.join(self.config.training.save_dir, 'checkpoints')
            checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')))
            if not checkpoints:
                raise ValueError("No checkpoints found!")
            self.last_good_checkpoint = checkpoints[-2]  # use the second last checkpoint
            
        print(f"Loading last good checkpoint: {self.last_good_checkpoint}")
        checkpoint = torch.load(self.last_good_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch']

def save_image(tensor, path):
    """save the image and add the debug information"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # convert the data from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2.0
        tensor = torch.clamp(tensor, 0, 1)  # ensure the value is in [0,1]
        
        # ensure the tensor does not need gradient
        img = tensor.detach().cpu()
        
        if img.ndim == 4:  # batch, channel, height, width
            img = img[0]  # only take the first image
        
        if img.ndim == 3 and img.shape[0] == 1:  # channel, height, width
            img = img.squeeze(0)  # remove the channel dimension
        
        # convert to numpy and adjust to 0-255 range
        img = (img.numpy() * 255).astype(np.uint8)
        
        # ensure the image size is reasonable
        if img.shape[0] > 10000 or img.shape[1] > 10000:
            logger.warning(f"Image size too large: {img.shape}, skipping save")
            return
            
        # save the image
        cv2.imwrite(str(path), img)
        logger.info(f"Successfully saved image to {path} with shape {img.shape}")
        
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        logger.error(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

if __name__ == "__main__":
    # load the config
    config = OmegaConf.load('E:\\DiffTSR-main\\configs\\two_stage_oracle_config.yaml')
    
    # verify the dataset
    if not verify_dataset(config.data.clean_train_dir, config.data.clean_val_dir):
        logger.error("Dataset verification failed!")
        exit(1)
    
    # create the trainer
    trainer = TwoStageTrainer(config)
    
    # start the training
    trainer.train() 