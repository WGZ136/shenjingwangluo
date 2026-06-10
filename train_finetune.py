"""
RAFT Fine-tuning Script for AirSim Blocks Dataset

This script fine-tunes a pre-trained RAFT model on custom AirSim Blocks data.
Features:
- Freezes feature encoder (fnet) parameters
- Trains correlation volume and update block (GRU)
- Uses multi-scale loss (sequence_loss)
- AdamW optimizer with cosine annealing scheduler
- TensorBoard logging
- Saves best model based on validation EPE
"""

from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import time
import numpy as np
from glob import glob
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

from raft import RAFT
from core.datasets import FlowDataset
from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor


# ============================================================================
# Dataset for AirSim Blocks Data
# ============================================================================

class AirSimBlocksDataset(FlowDataset):
    """
    Dataset for AirSim Blocks optical flow data.
    
    Expected directory structure:
        root/
            train/
                rgb/
                    frame_XXXXXX.png
                flow/
                    flow_XXXXXX.flo
            val/
                rgb/
                    frame_XXXXXX.png
                flow/
                    flow_XXXXXX.flo
            test/
                rgb/
                    frame_XXXXXX.png
                flow/
                    flow_XXXXXX.flo
    
    flow_XXXXXX.flo represents optical flow from frame_XXXXXX.png to frame_XXXXXX+1.png
    """
    
    def __init__(self, root, split='train', aug_params=None):
        super(AirSimBlocksDataset, self).__init__(aug_params)
        
        self.root = root
        self.split = split
        
        # Build paths
        rgb_dir = osp.join(root, split, 'rgb')
        flow_dir = osp.join(root, split, 'flow')
        
        if not osp.exists(rgb_dir):
            raise ValueError(f"RGB directory not found: {rgb_dir}")
        if not osp.exists(flow_dir):
            raise ValueError(f"Flow directory not found: {flow_dir}")
        
        # Find all flow files
        flow_files = sorted(glob(osp.join(flow_dir, 'flow_*.flo')))
        
        for flow_path in flow_files:
            # Extract frame number from flow filename (e.g., flow_000001.flo -> 000001)
            flow_name = osp.basename(flow_path)
            # Remove 'flow_' prefix and '.flo' suffix
            frame_num = flow_name.replace('flow_', '').replace('.flo', '')
            
            # Construct corresponding image paths
            frame1_path = osp.join(rgb_dir, f'frame_{frame_num}.png')
            
            # Next frame number (increment by 1, keeping same padding)
            next_num = str(int(frame_num) + 1).zfill(len(frame_num))
            frame2_path = osp.join(rgb_dir, f'frame_{next_num}.png')
            
            # Verify files exist
            if osp.exists(frame1_path) and osp.exists(frame2_path) and osp.exists(flow_path):
                self.image_list.append([frame1_path, frame2_path])
                self.flow_list.append(flow_path)
                self.extra_info.append((split, frame_num))
            else:
                if not osp.exists(frame1_path):
                    print(f"  Warning: Missing frame1: {frame1_path}")
                if not osp.exists(frame2_path):
                    print(f"  Warning: Missing frame2: {frame2_path}")
        
        print(f"[AirSimBlocksDataset] {split}: Found {len(self.flow_list)} samples")


# ============================================================================
# Loss Function (from original RAFT)
# ============================================================================

MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """
    Multi-scale loss function defined over sequence of flow predictions.
    
    Args:
        flow_preds: List of flow predictions at each iteration
        flow_gt: Ground truth flow
        valid: Valid mask
        gamma: Weight decay factor for earlier predictions
        max_flow: Maximum flow magnitude to consider
    """
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    # Exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


# ============================================================================
# Model Freezing Utilities
# ============================================================================

def freeze_feature_encoder(model):
    """
    Freeze the feature encoder (fnet) parameters.
    Only correlation volume builder and update block (GRU) will be trained.
    """
    # Freeze feature encoder
    for param in model.fnet.parameters():
        param.requires_grad = False
    
    print("[Freeze] Feature encoder (fnet) parameters frozen")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Freeze] Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.1f}%)")
    
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Optimizer and Scheduler
# ============================================================================

def fetch_optimizer(args, model):
    """
    Create the optimizer and learning rate scheduler.
    Uses AdamW with cosine annealing.
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wdecay,
        eps=args.epsilon
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_steps,
        eta_min=args.lr * 0.01  # Minimum LR is 1% of initial
    )

    return optimizer, scheduler


# ============================================================================
# Validation Function
# ============================================================================

@torch.no_grad()
def validate(model, val_loader, args):
    """
    Validate the model on validation set.
    Returns average EPE and other metrics.
    """
    model.eval()
    
    epe_list = []
    px1_list = []
    px3_list = []
    px5_list = []
    
    for i_batch, data_blob in enumerate(val_loader):
        image1, image2, flow, valid = [x.cuda() for x in data_blob]
        
        # Forward pass
        flow_predictions = model(image1, image2, iters=args.iters, test_mode=False)
        
        # Calculate EPE on final prediction
        flow_pred = flow_predictions[-1]
        
        # Compute EPE
        epe = torch.sum((flow_pred - flow)**2, dim=1).sqrt()
        
        # Filter valid pixels
        mag = torch.sum(flow**2, dim=1).sqrt()
        valid_mask = (valid >= 0.5) & (mag < MAX_FLOW)
        
        epe_valid = epe[valid_mask]
        
        if epe_valid.numel() > 0:
            epe_list.append(epe_valid.mean().item())
            px1_list.append((epe_valid < 1).float().mean().item())
            px3_list.append((epe_valid < 3).float().mean().item())
            px5_list.append((epe_valid < 5).float().mean().item())
    
    model.train()
    
    results = {
        'epe': np.mean(epe_list) if epe_list else float('inf'),
        '1px': np.mean(px1_list) if px1_list else 0,
        '3px': np.mean(px3_list) if px3_list else 0,
        '5px': np.mean(px5_list) if px5_list else 0,
    }
    
    return results


# ============================================================================
# Training Function
# ============================================================================

def train(args):
    """Main training function."""
    
    # Create model
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print(f"Parameter Count: {count_parameters(model):,}")
    
    # Load pretrained checkpoint
    if args.pretrained_ckpt is not None:
        print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        model.load_state_dict(torch.load(args.pretrained_ckpt), strict=False)
        print("Checkpoint loaded successfully")
    else:
        print("WARNING: No pretrained checkpoint provided, training from scratch!")
    
    model.cuda()
    model.train()
    
    # Freeze feature encoder
    freeze_feature_encoder(model.module)
    
    # Freeze batch norm layers (important for fine-tuning)
    model.module.freeze_bn()
    
    # Create datasets
    # Adjust crop size if image is smaller than crop size
    # Get actual image size from first training sample
    temp_dataset = AirSimBlocksDataset(
        root=args.data_root,
        split='train',
        aug_params=None
    )
    if len(temp_dataset) > 0:
        img1, _, _, _ = temp_dataset[0]
        actual_h, actual_w = img1.shape[1], img1.shape[2]
        crop_h = min(args.crop_size[0], actual_h)
        crop_w = min(args.crop_size[1], actual_w)
        print(f"Actual image size: {actual_h}x{actual_w}, Crop size: {crop_h}x{crop_w}")
    else:
        crop_h, crop_w = args.crop_size
    
    aug_params = {
        'crop_size': (crop_h, crop_w),
        'min_scale': -0.2,
        'max_scale': 0.6,
        'do_flip': True
    }
    
    train_dataset = AirSimBlocksDataset(
        root=args.data_root,
        split='train',
        aug_params=aug_params
    )
    
    val_dataset = AirSimBlocksDataset(
        root=args.data_root,
        split='val',
        aug_params=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Calculate total steps
    steps_per_epoch = len(train_loader)
    args.num_steps = steps_per_epoch * args.epochs
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {args.num_steps}")
    
    # Create optimizer and scheduler
    optimizer, scheduler = fetch_optimizer(args, model)
    
    # Mixed precision training
    scaler = GradScaler(enabled=args.mixed_precision)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training state
    total_steps = 0
    best_epe = float('inf')
    epoch = 0
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    while epoch < args.epochs:
        epoch += 1
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Training loop
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            
            image1, image2, flow, valid = [x.cuda() for x in data_blob]
            
            # Add noise for data augmentation (from original RAFT)
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
            
            # Forward pass
            flow_predictions = model(image1, image2, iters=args.iters)
            
            # Compute loss
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            
            # Update statistics
            total_steps += 1
            epoch_loss += loss.item()
            epoch_steps += 1
            
            # Log to TensorBoard
            writer.add_scalar('train/loss', loss.item(), total_steps)
            writer.add_scalar('train/epe', metrics['epe'], total_steps)
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], total_steps)
            
            # Print every LOG_FREQ iterations
            if total_steps % args.log_freq == 0:
                avg_loss = epoch_loss / epoch_steps
                print(f"[Step {total_steps}] Loss: {avg_loss:.4f}, "
                      f"EPE: {metrics['epe']:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Log detailed metrics less frequently
            if total_steps % (args.log_freq * 2) == 0:
                writer.add_scalar('train/1px', metrics['1px'], total_steps)
                writer.add_scalar('train/3px', metrics['3px'], total_steps)
                writer.add_scalar('train/5px', metrics['5px'], total_steps)
        
        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        print(f"Epoch {epoch} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Validation every VAL_FREQ epochs
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            print(f"\nRunning validation...")
            val_results = validate(model, val_loader, args)
            
            print(f"Validation Results:")
            print(f"  EPE: {val_results['epe']:.4f}")
            print(f"  1px: {val_results['1px']:.4f}")
            print(f"  3px: {val_results['3px']:.4f}")
            print(f"  5px: {val_results['5px']:.4f}")
            
            # Log validation metrics
            writer.add_scalar('val/epe', val_results['epe'], total_steps)
            writer.add_scalar('val/1px', val_results['1px'], total_steps)
            writer.add_scalar('val/3px', val_results['3px'], total_steps)
            writer.add_scalar('val/5px', val_results['5px'], total_steps)
            
            # Save best model
            if val_results['epe'] < best_epe:
                best_epe = val_results['epe']
                best_path = osp.join(args.checkpoint_dir, 'raft-finetuned-blocks-best.pth')
                torch.save(model.state_dict(), best_path)
                print(f"New best model saved! EPE: {best_epe:.4f}")
            
            # Save checkpoint
            checkpoint_path = osp.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_epe': best_epe,
                'val_results': val_results,
            }, checkpoint_path)
            
            model.train()
            model.module.freeze_bn()
    
    # Save final model
    final_path = osp.join(args.checkpoint_dir, 'raft-finetuned-blocks.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to: {final_path}")
    print(f"Best model EPE: {best_epe:.4f}")
    
    writer.close()
    print("Training completed!")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RAFT Fine-tuning on AirSim Blocks Dataset')
    
    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing train/val/test folders')
    parser.add_argument('--pretrained_ckpt', type=str, required=True,
                        help='Path to pretrained checkpoint (e.g., raft-things.pth)')
    
    # Model arguments
    parser.add_argument('--small', action='store_true',
                        help='Use small model variant')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='Use alternative correlation implementation')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--wdecay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--epsilon', type=float, default=1e-8,
                        help='Adam epsilon')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Loss weight decay factor')
    parser.add_argument('--iters', type=int, default=12,
                        help='Number of refinement iterations')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[368, 496],
                        help='Crop size for training images (H, W)')
    parser.add_argument('--add_noise', action='store_true', default=True,
                        help='Add noise to images for augmentation')
    
    # Logging and checkpointing
    parser.add_argument('--log_freq', type=int, default=500,
                        help='Print/log every N iterations')
    parser.add_argument('--val_freq', type=int, default=2,
                        help='Validate every N epochs')
    parser.add_argument('--log_dir', type=str, default='runs/raft_finetune',
                        help='TensorBoard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # System arguments
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='GPU IDs to use')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("RAFT Fine-tuning Configuration")
    print("=" * 60)
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 60)
    
    train(args)
