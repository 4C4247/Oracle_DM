import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse

from models.stroke_extractor import StrokeExtractor, StrokeLoss
from utils.oracle_dataset import OracleDataset

def visualize_batch(inputs, outputs, targets, epoch, save_dir):
    """result visualization"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'epoch_{epoch+1}_progress.png')
    
    try:
        # select the first sample for visualization
        input_img = inputs[0].cpu().detach().numpy()
        output_img = outputs[0].cpu().detach().numpy()
        target_img = targets[0].cpu().detach().numpy()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.title('Input')
        plt.imshow(input_img[0], cmap='gray')
        plt.axis('off')
        
        plt.subplot(132)
        plt.title('Predicted')
        plt.imshow(output_img[0], cmap='gray')
        plt.axis('off')
        
        plt.subplot(133)
        plt.title('Target')
        plt.imshow(target_img[0], cmap='gray')
        plt.axis('off')
        
        plt.savefig(save_path)
        plt.close()
        
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")

def save_checkpoint(model, optimizer, scheduler, epoch, save_dir, is_best=False):
    """save checkpoint safely"""
    try:
        # save to temporary file first
        temp_path = os.path.join(save_dir, 'temp_checkpoint.pth')
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, temp_path)
        
        # if saved successfully, rename to official file
        final_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        os.replace(temp_path, final_path)
        
        if is_best:
            best_path = os.path.join(save_dir, 'best_checkpoint.pth')
            import shutil
            shutil.copy2(final_path, best_path)
            
        print(f"Successfully saved checkpoint for epoch {epoch}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {str(e)}")

def verify_checkpoint(checkpoint_path):
    """verify checkpoint file is complete"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'epoch']
        
        for key in required_keys:
            if key not in checkpoint:
                return False
                
        return True
    except:
        return False

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir, start_epoch=0):
    try:
        model = model.to(device)
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # create visualization save directory
        vis_dir = os.path.join(save_dir, 'training_progress')
        os.makedirs(vis_dir, exist_ok=True)
        
        for epoch in range(start_epoch, num_epochs):
            # training phase
            model.train()
            train_loss = 0
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            
            for batch_idx, (images, targets) in enumerate(train_bar):
                try:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    
                    # calculate loss
                    loss = criterion(outputs, targets)  # now return scalar
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    
                    # update progress bar, show detailed loss
                    train_bar.set_postfix({
                        'total_loss': f'{loss.item():.4f}',
                        **{k: f'{v:.4f}' for k, v in criterion.loss_values.items()}
                    })
                    
                    # save visualization result every 100 batches
                    if batch_idx % 100 == 0:
                        visualize_batch(images, outputs, targets, epoch, vis_dir)
                        print(f"Saved visualization at epoch {epoch+1}, batch {batch_idx}")
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # validation phase
            model.eval()
            val_loss = 0
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            
            with torch.no_grad():
                for images, targets in val_bar:
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # update learning rate
            scheduler.step(avg_val_loss)
            
            # save latest_checkpoint after each epoch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }
            
            # save latest_checkpoint
            latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
            torch.save(checkpoint, latest_path)
            
            # save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint at epoch {epoch+1}")
                
                # random select an image for visualization
                with torch.no_grad():
                    # random select an image from training set
                    rand_idx = torch.randint(0, len(train_dataset), (1,)).item()
                    test_image, test_target = train_dataset[rand_idx]
                    test_image = test_image.unsqueeze(0).to(device)
                    test_target = test_target.unsqueeze(0).to(device)
                    
                    # generate prediction result
                    test_output = model(test_image)
                    
                    # save visualization result
                    visualize_batch(test_image, test_output, test_target, epoch, vis_dir)
                    print(f"Saved visualization at epoch {epoch+1} with random image {rand_idx}")
            
            # print training information
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Training Loss: {avg_train_loss:.6f}')
            print(f'Validation Loss: {avg_val_loss:.6f}')
        
        # save final checkpoint
        checkpoint_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved final checkpoint at epoch {epoch+1}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving latest checkpoint...")
        # save checkpoint when interrupted
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
        latest_path = os.path.join(save_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint at epoch {epoch}")
        raise  # rethrow interrupt exception

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, 
                      help='specify checkpoint path, e.g.: checkpoints/specific_checkpoint.pth')
    args = parser.parse_args()
    
    # create necessary directories
    save_dir = ''  # modify to your path
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('', exist_ok=True)
    
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # set random seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    # load dataset
    dataset = OracleDataset('')
    
    # split training set and validation set
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # initialize model and training components
    model = StrokeExtractor(in_channels=1).to(device)
    criterion = StrokeLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10,
        verbose=True,
        min_lr=1e-6
    )
    
    # specify checkpoint path
    checkpoint_path = None  # modify here to load different checkpoints
    start_epoch = 0
    
    try:
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print("No checkpoint found, starting from scratch")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Starting from scratch...")
    
    # start training
    print(f"Starting training from epoch {start_epoch}...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=100,
        device=device,
        save_dir=save_dir,
        start_epoch=start_epoch
    )
    print("Training completed!") 