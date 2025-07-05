import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import logging
import sys

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
    log_filename = os.path.join(log_dir, f'transfer_learning_{timestamp}.log')
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger()
    return logger, log_filename

def get_args_parser():
    parser = argparse.ArgumentParser('MAE transfer learning (Black & White version)', add_help=False)
    parser.add_argument('--total_epochs', default=60, type=int,
                        help='Total number of epochs for both approaches')
    parser.add_argument('--source_epochs', default=40, type=int,
                        help='Number of epochs to train source models')
    parser.add_argument('--target_epochs', default=20, type=int,
                        help='Number of epochs for fine-tuning')
    parser.add_argument('--scratch_epochs', default=60, type=int,
                        help='Number of epochs to train scratch models (should equal total_epochs)')

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

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

    parser.add_argument('--output_dir', default='./output_transfer_mae',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_transfer_mae',
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

    # Transfer learning specific parameters
    parser.add_argument('--source', type=int, default=0,
                        help='Source complexity level (1-7), 0 means all sources')
    parser.add_argument('--target', type=int, default=0,
                        help='Target complexity level (1-7), 0 means all targets')
    parser.add_argument('--skip_source_training', action='store_true',
                        help='Skip training source models and use existing ones')
    parser.add_argument('--skip_finetuning', action='store_true',
                        help='Skip fine-tuning and only train source models')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip evaluation of models')
    parser.add_argument('--force_reevaluation', action='store_true',
                        help='Force re-evaluation of already evaluated transfers')
    parser.set_defaults(force_reevaluation=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def evaluate_model_on_complexity(model, args, complexity_level):
    """Evaluate model on a specific complexity level using IoU and Dice score"""
    device = torch.device(args.device)
    model.eval()
    
    # Prepare dataset path - use validation data instead of training data
    if complexity_level > 0:
        data_path = os.path.join(args.data_path, 'val', f'tangrams_{complexity_level}_piece')
    else:
        data_path = os.path.join(args.data_path, 'val')
    
    # Simple transformation
    transform_eval = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    
    # Create dataset
    dataset_eval = TangramDatasetBW(data_path, transform=transform_eval)
    print(f"Evaluation dataset path: {data_path}")
    print(f"Evaluation dataset size: {len(dataset_eval)}")
    
    # Create data loader
    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )
    
    # Metrics
    total_iou = 0
    total_dice = 0
    total_samples = 0
    
    with torch.no_grad():
        for samples in data_loader_eval:
            # Handle the case where samples might be a list
            if isinstance(samples, list):
                # If it's a list, take the first element which should be the image tensor
                samples = samples[0]
            
            # Move to device
            samples = samples.to(device, non_blocking=True)
            
            # Forward pass
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio)
            
            # Unpatchify predictions
            pred = model.unpatchify(pred)
            
            # Apply sigmoid to get probabilities
            pred = torch.sigmoid(pred)
            
            # Threshold to get binary predictions
            pred_binary = (pred > 0.5).float()
            
            # Calculate IoU and Dice for each sample in the batch
            for i in range(samples.shape[0]):
                # Get ground truth and prediction
                gt = samples[i, 0]  # [H, W]
                pred_mask = pred_binary[i, 0]  # [H, W]
                
                # Calculate IoU
                intersection = (gt * pred_mask).sum()
                union = gt.sum() + pred_mask.sum() - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                
                # Calculate Dice
                dice = (2 * intersection + 1e-6) / (gt.sum() + pred_mask.sum() + 1e-6)
                
                # Accumulate
                total_iou += iou.item()
                total_dice += dice.item()
                total_samples += 1
    
    # Calculate average metrics
    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples
    
    print(f"Evaluation on complexity {complexity_level}: IoU={avg_iou:.4f}, Dice={avg_dice:.4f}")
    
    return avg_iou, avg_dice

def train_model(args, complexity_level, is_source=True, source_model_path=None, transfer_name=None, is_scratch=False):
    """Train a model on a specific complexity level"""
    # Create dataset path based on complexity level
    if complexity_level > 0:
        data_path = os.path.join(args.data_path, 'train', f'tangrams_{complexity_level}_piece')
    else:
        data_path = os.path.join(args.data_path, 'train')
    
    # Simple augmentation
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    
    dataset_train = TangramDatasetBW(data_path, transform=transform_train)
    print(f"Dataset path: {data_path}")
    print(f"Dataset size: {len(dataset_train)}")
    if len(dataset_train) == 0:
        raise ValueError(f"Dataset at {data_path} contains 0 images. Check path and image formats.")
    
    # Calculate global pos_weight
    white_pixels = 0
    total_pixels = 0
    for img in dataset_train:
        tensor = img if isinstance(img, torch.Tensor) else img[0]
        white_pixels += tensor.sum().item()
        total_pixels += tensor.numel()

    black_pixels = total_pixels - white_pixels
    eps = 1e-6
    pos_weight = black_pixels / (white_pixels + eps)
    print(f"Global pos_weight = {pos_weight:.3f}")
    
    # Create data loader
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # Define the model
    model = models_mae_bw.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
    model.global_pos_weight = torch.tensor([pos_weight], device=args.device)
    
    # Load source model if fine-tuning
    if not is_source and source_model_path:
        print(f"Loading source model from {source_model_path}")
        model.load_state_dict(torch.load(source_model_path, map_location=args.device))
    
    # Create proper device object
    device = torch.device(args.device)
    model.to(device)
    print("Model = %s" % str(model))
    
    # Set up optimizer
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    
    # Following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    
    # Set number of epochs based on training type
    if is_scratch:
        num_epochs = args.scratch_epochs
    else:
        num_epochs = args.source_epochs if is_source else args.target_epochs
    
    # Add epochs attribute to args for the learning rate scheduler
    args.epochs = num_epochs
    
    # Training loop
    print(f"Start training for {num_epochs} epochs")
    start_time = time.time()
    best_loss = float('inf')
    
    for epoch in range(args.start_epoch, num_epochs):
        # Create a wrapper for the data loader to handle list samples
        def process_samples(samples):
            if isinstance(samples, list):
                return samples[0]
            return samples
        
        # Create a custom data loader that returns tuples with two elements
        class CustomDataLoader:
            def __init__(self, data_loader):
                self.data_loader = data_loader
                self.iterator = None
            
            def __iter__(self):
                self.iterator = iter(self.data_loader)
                return self
            
            def __next__(self):
                try:
                    samples = next(self.iterator)
                    # Process samples if needed
                    if isinstance(samples, list):
                        samples = samples[0]
                    # Return a tuple with two elements as expected by train_one_epoch
                    return (samples, None)
                except StopIteration:
                    raise StopIteration
            
            def __len__(self):
                # Return the length of the original data loader
                return len(self.data_loader)
        
        # Use the custom data loader
        custom_loader = CustomDataLoader(data_loader_train)
        
        train_stats = train_one_epoch(
            model, custom_loader,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            args=args
        )
        
        # Remove best model tracking and saving
        # if train_stats['loss'] < best_loss:
        #     best_loss = train_stats['loss']
        #     if args.output_dir:
        #         best_model_path = os.path.join(args.output_dir, f"{'source' if is_source else 'transfer'}_model_complexity_{complexity_level}_best.pth")
        #         torch.save(model.state_dict(), best_model_path)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    # Save only the final model if transfer_name is provided
    if transfer_name is not None and args.output_dir:
        final_model_path = os.path.join(args.output_dir, transfer_name)
        torch.save(model.state_dict(), final_model_path)
    
    # Evaluate the model if not skipped
    iou = None
    dice = None
    if not args.skip_evaluation:
        print(f"Evaluating {'source' if is_source else 'fine-tuned'} model on complexity {complexity_level}")
        iou, dice = evaluate_model_on_complexity(model, args, complexity_level)
    
    return model, best_loss, total_time, iou, dice

def set_global_pos_weight(model, args, complexity_level):
    if complexity_level > 0:
        data_path = os.path.join(args.data_path, 'train', f'tangrams_{complexity_level}_piece')
    else:
        data_path = os.path.join(args.data_path, 'train')
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])
    dataset = TangramDatasetBW(data_path, transform=transform)
    white_pixels = 0
    total_pixels = 0
    for img in dataset:
        tensor = img if isinstance(img, torch.Tensor) else img[0]
        white_pixels += tensor.sum().item()
        total_pixels += tensor.numel()
    black_pixels = total_pixels - white_pixels
    eps = 1e-6
    pos_weight = black_pixels / (white_pixels + eps)
    model.global_pos_weight = torch.tensor([pos_weight], device=args.device)

def load_transfer_results(output_dir):
    """Load existing transfer results if available."""
    results_path = os.path.join(output_dir, "transfer_results.json")
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}

def save_transfer_results(results, output_dir):
    """Save transfer results to file."""
    results_path = os.path.join(output_dir, "transfer_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Updated transfer results saved to {results_path}")

def main(args):
    # Set up logging
    logger, log_filename = setup_logging(log_dir=args.output_dir)
    
    # Redirect stdout and stderr to logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    
    # Log start
    logging.info("=" * 80)
    logging.info("TRANSFER LEARNING EXPERIMENT - MAE MODEL")
    logging.info("=" * 80)
    logging.info(f"Command line arguments: {args}")
    
    # Create output directory
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine source and target complexity levels
    source_levels = [args.source] if args.source > 0 else list(range(1, 8))
    target_levels = [args.target] if args.target > 0 else list(range(1, 8))
    
    # Initialize results dictionary
    results = {}
    scratch_results = {}
    
    try:
        # First, train source models for each complexity level if needed
        if not args.skip_source_training:
            logging.info("Training source models for each complexity level")
            source_results = {}
            
            for source in source_levels:
                source_model_path = os.path.join(args.output_dir, f"source_model_complexity_{source}.pth")
                
                # Skip if model already exists
                if os.path.exists(source_model_path):
                    logging.info(f"Source model for complexity {source} already exists at {source_model_path}")
                    
                    # Load the model for evaluation
                    model = models_mae_bw.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
                    model.load_state_dict(torch.load(source_model_path, map_location=args.device))
                    set_global_pos_weight(model, args, source)
                    model.to(args.device)
                    
                    # Evaluate if not skipped
                    iou = None
                    dice = None
                    if not args.skip_evaluation:
                        iou, dice = evaluate_model_on_complexity(model, args, source)
                    
                    source_results[source] = {
                        "status": "existing", 
                        "path": source_model_path,
                        "iou": float(iou) if iou is not None else None,
                        "dice": float(dice) if dice is not None else None
                    }
                    continue
                
                logging.info(f"Training source model for complexity {source}")
                
                # Create a copy of args for this training run
                train_args = argparse.Namespace(**vars(args))
                # No need to set epochs here as train_model will use source_epochs
                
                # Train the model
                model, best_loss, training_time, iou, dice = train_model(train_args, source, is_source=True)
                
                # Save the model
                torch.save(model.state_dict(), source_model_path)
                
                # Record results
                source_results[source] = {
                    "status": "trained",
                    "path": source_model_path,
                    "best_loss": float(best_loss),
                    "training_time": training_time,
                    "iou": float(iou) if iou is not None else None,
                    "dice": float(dice) if dice is not None else None
                }
                logging.info(f"Source model for complexity {source} trained with best loss: {best_loss:.4f}")
                if iou is not None:
                    logging.info(f"Source model for complexity {source} evaluated with IoU: {iou:.4f}, Dice: {dice:.4f}")
            
            # Save source results
            with open(os.path.join(args.output_dir, "source_results.json"), 'w') as f:
                json.dump(source_results, f, indent=2)
        else:
            # Load existing source results
            source_results_path = os.path.join(args.output_dir, "source_results.json")
            if os.path.exists(source_results_path):
                with open(source_results_path, 'r') as f:
                    source_results = json.load(f)
                logging.info("Loaded existing source model results")
            else:
                source_results = {}
                for source in source_levels:
                    source_model_path = os.path.join(args.output_dir, f"source_model_complexity_{source}.pth")
                    if os.path.exists(source_model_path):
                        # Load the model for evaluation
                        model = models_mae_bw.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
                        model.load_state_dict(torch.load(source_model_path, map_location=args.device))
                        set_global_pos_weight(model, args, source)
                        model.to(args.device)
                        
                        # Evaluate if not skipped
                        iou = None
                        dice = None
                        if not args.skip_evaluation:
                            iou, dice = evaluate_model_on_complexity(model, args, source)
                        
                        source_results[source] = {
                            "status": "existing", 
                            "path": source_model_path,
                            "iou": float(iou) if iou is not None else None,
                            "dice": float(dice) if dice is not None else None
                        }
        
        # Train scratch models for each target complexity if needed
        if not os.path.exists(os.path.join(args.output_dir, "scratch_results.json")):
            logging.info("Training scratch models for baseline comparison")
            
            for target in target_levels:
                logging.info(f"Training scratch model for complexity {target}")
                
                # Create a copy of args for this training run
                train_args = argparse.Namespace(**vars(args))
                # Set epochs to scratch_epochs for scratch models
                train_args.source_epochs = args.scratch_epochs
                
                # Train a new model from scratch
                model, best_loss, training_time, iou, dice = train_model(train_args, target, is_source=False, is_scratch=True)
                
                # Save the model with a distinct scratch model path
                scratch_model_path = os.path.join(args.output_dir, f"scratch_model_complexity_{target}.pth")
                torch.save(model.state_dict(), scratch_model_path)
                
                # Record results
                scratch_results[target] = {
                    "status": "trained",
                    "path": scratch_model_path,
                    "best_loss": float(best_loss),
                    "training_time": training_time,
                    "iou": float(iou) if iou is not None else None,
                    "dice": float(dice) if dice is not None else None
                }
                logging.info(f"Scratch model for complexity {target} trained with best loss: {best_loss:.4f}")
                if iou is not None:
                    logging.info(f"Scratch model for complexity {target} evaluated with IoU: {iou:.4f}, Dice: {dice:.4f}")
            
            # Save scratch results
            with open(os.path.join(args.output_dir, "scratch_results.json"), 'w') as f:
                json.dump(scratch_results, f, indent=2)
        else:
            # Load existing scratch results
            with open(os.path.join(args.output_dir, "scratch_results.json"), 'r') as f:
                scratch_results = json.load(f)
            logging.info("Loaded existing scratch model results")
        
        # Load existing transfer results
        results = load_transfer_results(args.output_dir)
        
        # Now fine-tune source models on target complexities
        if not args.skip_finetuning:
            for source in source_levels:
                for target in target_levels:
                    # Skip self-transfer
                    if source == target:
                        logging.info(f"Skipping self-transfer from {source} to {source}")
                        continue
                    
                    # Skip if already evaluated
                    transfer_key = f"{source}to{target}"
                    if transfer_key in results and not args.force_reevaluation:
                        logging.info(f"Transfer {source}→{target} already evaluated, skipping...")
                        continue
                    
                    logging.info(f"Fine-tuning from source complexity {source} to target complexity {target}")
                    start_time = time.time()
                    
                    # Load source model
                    source_model_path = os.path.join(args.output_dir, f"source_model_complexity_{source}.pth")
                    if not os.path.exists(source_model_path):
                        logging.error(f"Source model for complexity {source} not found at {source_model_path}")
                        continue
                    
                    # Create a copy of args for this training run
                    train_args = argparse.Namespace(**vars(args))
                    
                    # Check if transfer model already exists
                    transfer_model_path = os.path.join(args.output_dir, f"transfer_{source}to{target}_model.pth")
                    if os.path.exists(transfer_model_path):
                        logging.info(f"Loading existing transfer model for {source}→{target}")
                        model = models_mae_bw.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)
                        model.load_state_dict(torch.load(transfer_model_path, map_location=args.device))
                        model.to(args.device)
                        set_global_pos_weight(model, args, target)
                        
                        # Evaluate existing model
                        iou, dice = evaluate_model_on_complexity(model, args, target) if not args.skip_evaluation else (None, None)
                        best_loss = 0  # Placeholder for existing model
                        training_time = 0  # Placeholder for existing model
                    else:
                        # Fine-tune the model
                        model, best_loss, training_time, iou, dice = train_model(
                            train_args, target, is_source=False, 
                            source_model_path=source_model_path,
                            transfer_name=f"transfer_{source}to{target}_model.pth"
                        )
                    
                    # Get scratch model metrics for comparison
                    scratch_iou = None
                    scratch_dice = None
                    if str(target) in scratch_results:
                        scratch_iou = scratch_results[str(target)].get("iou")
                        scratch_dice = scratch_results[str(target)].get("dice")
                    
                    # Calculate transfer benefit
                    transfer_benefit_iou = None
                    transfer_benefit_dice = None
                    if iou is not None and scratch_iou is not None:
                        transfer_benefit_iou = float(iou) - float(scratch_iou)
                    if dice is not None and scratch_dice is not None:
                        transfer_benefit_dice = float(dice) - float(scratch_dice)
                    
                    # Save transfer results
                    transfer_result = {
                        "source": source,
                        "target": target,
                        "source_model_path": source_model_path,
                        "transfer_model_path": transfer_model_path,
                        "best_loss": float(best_loss),
                        "training_time": training_time,
                        "iou": float(iou) if iou is not None else None,
                        "dice": float(dice) if dice is not None else None,
                        "scratch_iou": float(scratch_iou) if scratch_iou is not None else None,
                        "scratch_dice": float(scratch_dice) if scratch_dice is not None else None,
                        "transfer_benefit_iou": float(transfer_benefit_iou) if transfer_benefit_iou is not None else None,
                        "transfer_benefit_dice": float(transfer_benefit_dice) if transfer_benefit_dice is not None else None,
                        "evaluation_timestamp": time.strftime("%Y%m%d_%H%M%S")
                    }
                    
                    results[transfer_key] = transfer_result
                    
                    # Save results after each transfer
                    save_transfer_results(results, args.output_dir)
                    
                    elapsed = time.time() - start_time
                    logging.info(f"Transfer {source}→{target} completed in {elapsed:.2f} seconds")
                    logging.info(f"Best loss: {best_loss:.4f}")
                    if iou is not None:
                        logging.info(f"Transfer model evaluated with IoU: {iou:.4f}, Dice: {dice:.4f}")
                        if transfer_benefit_iou is not None:
                            logging.info(f"Transfer benefit - IoU: {transfer_benefit_iou:.4f}, Dice: {transfer_benefit_dice:.4f}")
        
        # Save final transfer matrix results
        with open(os.path.join(args.output_dir, "transfer_results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Saved transfer results to {os.path.join(args.output_dir, 'transfer_results.json')}")
        
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
    
    finally:
        # Reset stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        logging.info("=" * 80)
        logging.info("TRANSFER LEARNING EXPERIMENT COMPLETED")
        logging.info("=" * 80)
        print(f"Execution completed. Log saved to {log_filename}")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 