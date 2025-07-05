import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging
import sys
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F

import timm
assert timm.__version__ == "0.3.2"
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_bw
from engine_pretrain import train_one_epoch
from util.datasets_bw import TangramDatasetBW, GRAYSCALE_MEAN, GRAYSCALE_STD

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message and not message.isspace():
            self.logger.log(self.level, message)

    def flush(self):
        pass

def setup_logging(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f'curriculum_learning_{timestamp}.log')
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create formatters and handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger, log_filename

def get_args_parser():
    parser = argparse.ArgumentParser('MAE curriculum learning (Black & White version)', add_help=False)
    parser.add_argument('--epochs_per_stage', default=30, type=int,
                        help='Number of epochs to train on each curriculum stage')
    parser.add_argument('--epochs', default=210, type=int,
                        help='Total number of epochs (7 stages * epochs_per_stage)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16_bw', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.25, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./mae/dataset_imagenet_style', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_curriculum_mae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_curriculum_mae',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='mps',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

def create_curriculum_dataset(args, current_stage, curriculum_type):
    """Create dataset for current curriculum stage"""
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    if curriculum_type == 'forward':
        # Forward curriculum: cumulative 1->1+2->1+2+3->1+2+3+4->1+2+3+4+5->1+2+3+4+5+6->1+2+3+4+5+6+7
        datasets = []
        # Add all datasets up to current stage
        for stage in range(1, current_stage + 1):
            data_path = os.path.join(args.data_path, 'train', f'tangrams_{stage}_piece')
            stage_dataset = TangramDatasetBW(data_path, transform=transform_train)
            datasets.append(stage_dataset)
            logging.info(f"Added stage {stage} dataset with {len(stage_dataset)} samples")
        
        # Combine all datasets
        dataset = torch.utils.data.ConcatDataset(datasets)
        logging.info(f"Combined dataset has {len(dataset)} total samples")
    
    elif curriculum_type == 'reverse':
        # Reverse curriculum: cumulative 7->7+6->7+6+5->7+6+5+4->7+6+5+4+3->7+6+5+4+3+2->7+6+5+4+3+2+1
        datasets = []
        # Add all datasets from highest to current
        for complexity in range(8 - current_stage, 8):
            data_path = os.path.join(args.data_path, 'train', f'tangrams_{complexity}_piece')
            stage_dataset = TangramDatasetBW(data_path, transform=transform_train)
            datasets.append(stage_dataset)
            logging.info(f"Added complexity {complexity} dataset with {len(stage_dataset)} samples")
        
        # Combine all datasets
        dataset = torch.utils.data.ConcatDataset(datasets)
        logging.info(f"Combined dataset has {len(dataset)} total samples")
    
    else:  # mixed
        # Mixed curriculum: cumulative based on predefined stages
        stages = [1, 7, 2, 6, 3, 5, 4]  # Original stage order
        datasets = []
        # Add all datasets up to current stage index
        for i in range(current_stage):
            complexity = stages[i]
            data_path = os.path.join(args.data_path, 'train', f'tangrams_{complexity}_piece')
            stage_dataset = TangramDatasetBW(data_path, transform=transform_train)
            datasets.append(stage_dataset)
            logging.info(f"Added complexity {complexity} dataset with {len(stage_dataset)} samples")
        
        # Combine all datasets
        dataset = torch.utils.data.ConcatDataset(datasets)
        logging.info(f"Combined dataset has {len(dataset)} total samples")
    
    return dataset

def compute_iou_and_dice(pred_bin: torch.Tensor,
                         gt_bin:   torch.Tensor,
                         eps:      float = 1e-8):
    """
    pred_bin, gt_bin: [M, P] where M=number of masked patches, P=patch_size**2.
    Returns the mean IoU and mean Dice across those M patches.
    """
    # Per-patch intersection & union
    inter  = (pred_bin * gt_bin).sum(dim=1)                             # [M]
    union  = (pred_bin + gt_bin - pred_bin * gt_bin).sum(dim=1)         # [M]
    ious   = inter / (union + eps)                                      # [M]

    # Per-patch Dice
    pred_sum = pred_bin.sum(dim=1)
    gt_sum   = gt_bin.sum(dim=1)
    dices    = 2.0 * inter / (pred_sum + gt_sum + eps)                  # [M]

    # Mean over patches
    return ious.mean().item(), dices.mean().item()

def evaluate_model_on_validation(model, args):
    """Evaluate model on validation set using patch-wise metrics"""
    device = torch.device(args.device)
    model.eval()
    
    # Prepare validation dataset
    transform_eval = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    
    # Create dataset for all validation data
    data_path = os.path.join(args.data_path, 'val')
    dataset_eval = TangramDatasetBW(data_path, transform=transform_eval)
    
    # Create data loader
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )
    
    # Metrics
    total_iou = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for samples in data_loader_eval:
            if isinstance(samples, list):
                samples = samples[0]
            
            samples = samples.to(device, non_blocking=True)
            
            # Forward pass
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
            
            # Get ground truth patches
            gt_patches = model.patchify(samples)  # [N, L, P]
            
            # Apply sigmoid to predictions
            pred_probs = torch.sigmoid(pred)  # [N, L, P]
            
            # Threshold to get binary predictions
            pred_binary = (pred_probs > 0.5).float()  # [N, L, P]
            
            # Flatten batch and patch dimensions for IoU calculation
            N, L, P = pred_binary.shape
            pred_flat = pred_binary.reshape(N*L, P)
            gt_flat = gt_patches.reshape(N*L, P)
            mask_flat = mask.reshape(N*L)
            
            # Select only masked patches
            sel = mask_flat.bool()
            pred_sel = pred_flat[sel]
            gt_sel = gt_flat[sel]
            
            # Calculate mean IoU and Dice for this batch's masked patches
            batch_iou, batch_dice = compute_iou_and_dice(pred_sel, gt_sel)
            
            # Update totals
            total_iou += batch_iou
            total_dice += batch_dice
            num_batches += 1
    
    # Calculate average metrics across batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches
    
    return avg_iou, avg_dice

def set_global_pos_weight(model, dataset, device):
    """Calculate and set global_pos_weight based on dataset statistics"""
    total_pixels = 0
    total_white_pixels = 0
    
    for data in dataset:
        # Handle different return types from dataset
        if isinstance(data, tuple):
            img = data[0]  # First element is the image
        elif isinstance(data, list):
            img = data[0]
        else:
            img = data
            
        # Convert to tensor if needed
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
            
        total_pixels += img.numel()
        total_white_pixels += (img > 0.5).sum().item()
    
    # Calculate ratio of black to white pixels
    black_pixels = total_pixels - total_white_pixels
    pos_weight = black_pixels / (total_white_pixels + 1e-6)  # Add epsilon for numerical stability
    model.global_pos_weight = torch.tensor([pos_weight], device=device)
    logging.info(f"Set global_pos_weight to {pos_weight:.4f} (black/white ratio)")

def train_curriculum(args, curriculum_type):
    """Train model using specified curriculum strategy"""
    try:
        # Set up logging
        logger, log_filename = setup_logging(log_dir=args.output_dir)
        logging.info(f"Log file created at: {log_filename}")
        
        # Make sure total epochs matches epochs_per_stage * 7
        args.epochs = args.epochs_per_stage * 7
        
        sys.stdout = LoggerWriter(logger, logging.INFO)
        sys.stderr = LoggerWriter(logger, logging.ERROR)
        
        logging.info(f"Starting {curriculum_type} curriculum training")
        
        # Create output directory
        if args.output_dir:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            logging.info(f"Created output directory: {args.output_dir}")
        
        # Initialize model
        logging.info("Creating model...")
        model = models_mae_bw.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
        device = torch.device(args.device)
        model.to(device)
        logging.info(f"Model created and moved to device: {args.device}")
        
        # Set up optimizer
        logging.info("Setting up optimizer...")
        eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
        if args.lr is None:
            args.lr = args.blr * eff_batch_size / 256
        
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()
        logging.info("Optimizer setup complete")
        
        # Training loop
        total_epochs = args.epochs_per_stage * 7  # 7 stages
        start_time = time.time()
        
        for stage in range(1, 8):  # 7 stages
            logging.info(f"Starting stage {stage} of {curriculum_type} curriculum")
            
            # Create dataset for current stage
            logging.info(f"Creating dataset for stage {stage}...")
            dataset_train = create_curriculum_dataset(args, stage, curriculum_type)
            logging.info(f"Dataset created with {len(dataset_train)} samples")
            
            # Set global_pos_weight for this stage's dataset
            logging.info("Calculating global_pos_weight...")
            set_global_pos_weight(model, dataset_train, device)
            
            # Create data loader
            logging.info("Creating data loader...")
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )
            logging.info("Data loader created")
            
            # Train for this stage
            for epoch in range(args.epochs_per_stage):
                logging.info(f"Starting epoch {epoch+1}/{args.epochs_per_stage}")
                train_stats = train_one_epoch(
                    model, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    log_writer=None,
                    args=args
                )
                logging.info(f"Stage {stage}, Epoch {epoch+1}/{args.epochs_per_stage}, Loss: {train_stats['loss']:.4f}")
        
        # Final evaluation on validation set
        final_iou, final_dice = evaluate_model_on_validation(model, args)
        logging.info(f"Final validation metrics - IoU: {final_iou:.4f}, Dice: {final_dice:.4f}")
        
        # Save final model
        if args.output_dir:
            final_model_path = os.path.join(args.output_dir, f"{curriculum_type}_model_final.pth")
            torch.save(model.state_dict(), final_model_path)
            logging.info(f"Saved final model to {final_model_path}")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f'Total training time: {total_time_str}')
        
        # Reset stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        return final_iou, final_dice
    except Exception as e:
        logging.error(f"Error in train_curriculum: {str(e)}")
        raise

def plot_curriculum_results(results, output_dir):
    """Plot IoU and Dice scores for all curriculum strategies"""
    # Extract data
    curricula = list(results.keys())
    iou_scores = [results[c]['iou'] for c in curricula]
    dice_scores = [results[c]['dice'] for c in curricula]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot IoU scores
    bars1 = ax1.bar(curricula, iou_scores, color=['#2ecc71', '#e74c3c', '#3498db'])
    ax1.set_title('IoU Scores by Curriculum Strategy')
    ax1.set_ylabel('IoU Score')
    ax1.set_ylim(0, 1)  # IoU is between 0 and 1
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Plot Dice scores
    bars2 = ax2.bar(curricula, dice_scores, color=['#2ecc71', '#e74c3c', '#3498db'])
    ax2.set_title('Dice Scores by Curriculum Strategy')
    ax2.set_ylabel('Dice Score')
    ax2.set_ylim(0, 1)  # Dice is between 0 and 1
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'curriculum_comparison.png'))
    plt.close()

def main(args):
    # Train with all three curriculum strategies
    curricula = ['forward', 'reverse', 'mixed']
    results = {}
    
    logging.info("Starting main execution")
    logging.info(f"Args: {args}")
    
    for curriculum in curricula:
        logging.info(f"\n{'='*80}\nStarting {curriculum} curriculum training\n{'='*80}")
        try:
            iou, dice = train_curriculum(args, curriculum)
            results[curriculum] = {
                'iou': float(iou),
                'dice': float(dice)
            }
        except Exception as e:
            logging.error(f"Error in {curriculum} curriculum: {str(e)}")
            raise

    # Save results
    if args.output_dir:
        results_path = os.path.join(args.output_dir, 'curriculum_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved curriculum results to {results_path}")
        
        # Plot and save comparison
        plot_curriculum_results(results, args.output_dir)
        logging.info(f"Saved comparison plot to {os.path.join(args.output_dir, 'curriculum_comparison.png')}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up logging once at the start
    logger, log_filename = setup_logging(args.output_dir)
    logging.info("Starting script with logging configured")
    
    try:
        main(args)
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise 