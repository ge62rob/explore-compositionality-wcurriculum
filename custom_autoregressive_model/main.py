import os
import argparse
import sys
import time
import config  # Import the module instead of just the variable
from training import train_forward_curriculum, train_reverse_curriculum, train_mixed_curriculum
from evaluation import evaluate_model_on_complexity
from utils import plot_results
import logging
from logging_utils import setup_logging, LoggerWriter

def main():
    # Set up logging first
    logger, log_filename = setup_logging()
    
    # Redirect stdout and stderr to logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)
    
    # Log start of program
    logging.info("=" * 80)
    logging.info("STARTING TANGRAM TRANSFORMER TRAINING")
    logging.info("=" * 80)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Autoregressive Transformer for Tangram Prediction')
    parser.add_argument('--dataset', type=str, default=config.DATASET_PATH, 
                        help='Path to dataset directory')
    parser.add_argument('--mode', type=str, choices=['all', 'forward', 'reverse', 'mixed'], 
                        default='all', help='Which curriculum to run')
    parser.add_argument('--eval_only', action='store_true', 
                        help='Only run evaluation on pre-trained models')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs and model checkpoints')
    args = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Log the arguments
    logging.info(f"Command line arguments: {args}")
    
    # Update dataset path if provided - modify the module attribute
    if args.dataset != config.DATASET_PATH:
        config.DATASET_PATH = args.dataset  # Update the module attribute directly
        logging.info(f"Using dataset path: {config.DATASET_PATH}")
    
    # Check if the dataset path exists
    if not os.path.exists(config.DATASET_PATH):
        logging.error(f"Error: Dataset path {config.DATASET_PATH} does not exist!")
        return
    
    # Log all subdirectories in the dataset
    dataset_folders = [f for f in os.listdir(config.DATASET_PATH) 
                        if os.path.isdir(os.path.join(config.DATASET_PATH, f))]
    logging.info(f"Found dataset folders: {dataset_folders}")
    
    # Initialize variables to store results
    forward_model, forward_losses = None, []
    reverse_model, reverse_losses = None, []
    mixed_model, mixed_losses = None, []
    
    # Log hardware information
    import torch
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logging.info(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    logging.info(f"Device being used: {config.DEVICE}")
    
    # Log model hyperparameters
    logging.info(f"Model configuration:")
    logging.info(f"  IMAGE_SIZE: {config.IMAGE_SIZE}")
    logging.info(f"  PATCH_SIZE: {config.PATCH_SIZE}")
    logging.info(f"  NUM_PATCHES: {config.NUM_PATCHES}")
    logging.info(f"  EMBED_DIM: {config.EMBED_DIM}")
    logging.info(f"  MASK_RATIO: {config.MASK_RATIO}")
    logging.info(f"  NUM_HEADS: {config.NUM_HEADS}")
    logging.info(f"  NUM_LAYERS: {config.NUM_LAYERS}")
    logging.info(f"  LEARNING_RATE: {config.LEARNING_RATE}")
    logging.info(f"  BATCH_SIZE: {config.BATCH_SIZE}")
    logging.info(f"  EPOCHS: {config.EPOCHS}")
    logging.info(f"  PATIENCE: {config.PATIENCE}")
    
    try:
        # Training phase
        if not args.eval_only:
            if args.mode in ['all', 'forward']:
                logging.info("Starting forward curriculum training...")
                start_time = time.time()
                forward_model, forward_losses = train_forward_curriculum(output_dir=args.output_dir)
                elapsed = time.time() - start_time
                logging.info(f"Forward curriculum training completed in {elapsed:.2f} seconds")
                # Save losses to CSV
                save_losses_to_csv(forward_losses, os.path.join(args.output_dir, 'forward_losses.csv'))
                
            if args.mode in ['all', 'reverse']:
                logging.info("Starting reverse curriculum training...")
                start_time = time.time()
                reverse_model, reverse_losses = train_reverse_curriculum(output_dir=args.output_dir)
                elapsed = time.time() - start_time
                logging.info(f"Reverse curriculum training completed in {elapsed:.2f} seconds")
                # Save losses to CSV
                save_losses_to_csv(reverse_losses, os.path.join(args.output_dir, 'reverse_losses.csv'))
                
            if args.mode in ['all', 'mixed']:
                logging.info("Starting mixed curriculum training...")
                start_time = time.time()
                mixed_model, mixed_losses = train_mixed_curriculum(output_dir=args.output_dir)
                elapsed = time.time() - start_time
                logging.info(f"Mixed curriculum training completed in {elapsed:.2f} seconds")
                # Save losses to CSV
                save_losses_to_csv(mixed_losses, os.path.join(args.output_dir, 'mixed_losses.csv'))
        else:
            # Load pre-trained models for evaluation
            from model import AutoregressiveTransformer
            import torch
            from config import DEVICE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, NUM_PATCHES
            
            logging.info("Evaluation only mode - loading pre-trained models...")
            
            if args.mode in ['all', 'forward']:
                model_path = os.path.join(args.output_dir, "best_autoregressive_transformer_forward.pth")
                if not os.path.exists(model_path):
                    model_path = "best_autoregressive_transformer_forward.pth"  # fallback to current dir
                
                if os.path.exists(model_path):
                    logging.info(f"Loading forward model from {model_path}")
                    forward_model = AutoregressiveTransformer(
                        embed_dim=EMBED_DIM,
                        num_heads=NUM_HEADS,
                        num_layers=NUM_LAYERS,
                        num_patches=NUM_PATCHES
                    ).to(DEVICE)
                    forward_model.load_state_dict(torch.load(model_path))
                else:
                    logging.warning(f"Model file {model_path} not found!")
                
            if args.mode in ['all', 'reverse']:
                model_path = os.path.join(args.output_dir, "best_autoregressive_transformer_reverse.pth")
                if not os.path.exists(model_path):
                    model_path = "best_autoregressive_transformer_reverse.pth"  # fallback to current dir
                
                if os.path.exists(model_path):
                    logging.info(f"Loading reverse model from {model_path}")
                    reverse_model = AutoregressiveTransformer(
                        embed_dim=EMBED_DIM,
                        num_heads=NUM_HEADS,
                        num_layers=NUM_LAYERS,
                        num_patches=NUM_PATCHES
                    ).to(DEVICE)
                    reverse_model.load_state_dict(torch.load(model_path))
                else:
                    logging.warning(f"Model file {model_path} not found!")
                
            if args.mode in ['all', 'mixed']:
                model_path = os.path.join(args.output_dir, "best_autoregressive_transformer_mixed.pth")
                if not os.path.exists(model_path):
                    model_path = "best_autoregressive_transformer_mixed.pth"  # fallback to current dir
                
                if os.path.exists(model_path):
                    logging.info(f"Loading mixed model from {model_path}")
                    mixed_model = AutoregressiveTransformer(
                        embed_dim=EMBED_DIM,
                        num_heads=NUM_HEADS,
                        num_layers=NUM_LAYERS,
                        num_patches=NUM_PATCHES
                    ).to(DEVICE)
                    mixed_model.load_state_dict(torch.load(model_path))
                else:
                    logging.warning(f"Model file {model_path} not found!")
        
        # Evaluation phase
        logging.info("Starting evaluation phase...")
        forward_scores = []
        reverse_scores = []
        mixed_scores = []
        
        all_scores = {
            'complexity': [],
            'forward_iou': [], 'forward_dice': [],
            'reverse_iou': [], 'reverse_dice': [],
            'mixed_iou': [], 'mixed_dice': []
        }
        
        for c in range(1, 8):
            folder_path = os.path.join(config.DATASET_PATH, f"tangrams_{c}_piece")
            all_scores['complexity'].append(c)
            
            if not os.path.exists(folder_path):
                logging.warning(f"Folder for complexity {c} not found at {folder_path}")
                # Add placeholder values
                all_scores['forward_iou'].append(None)
                all_scores['forward_dice'].append(None)
                all_scores['reverse_iou'].append(None)
                all_scores['reverse_dice'].append(None)
                all_scores['mixed_iou'].append(None)
                all_scores['mixed_dice'].append(None)
                continue
            
            if forward_model and os.path.exists(folder_path):
                logging.info(f"Evaluating forward model on complexity {c}...")
                f_iou, f_dice = evaluate_model_on_complexity(forward_model, folder_path)
                forward_scores.append((f_iou, f_dice))
                all_scores['forward_iou'].append(f_iou)
                all_scores['forward_dice'].append(f_dice)
                logging.info(f"Complexity {c}: Forward(IoU={f_iou:.4f}, Dice={f_dice:.4f})")
            else:
                all_scores['forward_iou'].append(None)
                all_scores['forward_dice'].append(None)
            
            if reverse_model and os.path.exists(folder_path):
                logging.info(f"Evaluating reverse model on complexity {c}...")
                r_iou, r_dice = evaluate_model_on_complexity(reverse_model, folder_path)
                reverse_scores.append((r_iou, r_dice))
                all_scores['reverse_iou'].append(r_iou)
                all_scores['reverse_dice'].append(r_dice)
                logging.info(f"Complexity {c}: Reverse(IoU={r_iou:.4f}, Dice={r_dice:.4f})")
            else:
                all_scores['reverse_iou'].append(None)
                all_scores['reverse_dice'].append(None)
            
            if mixed_model and os.path.exists(folder_path):
                logging.info(f"Evaluating mixed model on complexity {c}...")
                m_iou, m_dice = evaluate_model_on_complexity(mixed_model, folder_path)
                mixed_scores.append((m_iou, m_dice))
                all_scores['mixed_iou'].append(m_iou)
                all_scores['mixed_dice'].append(m_dice)
                logging.info(f"Complexity {c}: Mixed(IoU={m_iou:.4f}, Dice={m_dice:.4f})")
            else:
                all_scores['mixed_iou'].append(None)
                all_scores['mixed_dice'].append(None)
        
        # Save evaluation results to CSV
        save_scores_to_csv(all_scores, os.path.join(args.output_dir, 'evaluation_results.csv'))
        
        # Plot results if we have data from all curriculum types
        if forward_scores and reverse_scores and mixed_scores:
            logging.info("Generating result plots...")
            plot_paths = plot_results(
                forward_losses, reverse_losses, mixed_losses,
                forward_scores, reverse_scores, mixed_scores,
                output_dir=args.output_dir
            )
            logging.info(f"Plots saved to: {plot_paths}")
    
    except Exception as e:
        logging.exception(f"An error occurred during execution: {e}")
    
    finally:
        # Log completion
        logging.info("=" * 80)
        logging.info("PROGRAM EXECUTION COMPLETED")
        logging.info("=" * 80)
        logging.info(f"Log file saved to: {log_filename}")
        
        # Reset stdout and stderr
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        
        print(f"Execution completed. Log saved to {log_filename}")

def save_losses_to_csv(losses, filename):
    """Save losses to a CSV file"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'loss'])
        for i, loss in enumerate(losses):
            writer.writerow([i+1, loss])
    logging.info(f"Losses saved to {filename}")

def save_scores_to_csv(scores_dict, filename):
    """Save evaluation scores to a CSV file"""
    import csv
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['complexity', 'forward_iou', 'forward_dice', 
                         'reverse_iou', 'reverse_dice', 
                         'mixed_iou', 'mixed_dice'])
        # Write data rows
        for i in range(len(scores_dict['complexity'])):
            writer.writerow([
                scores_dict['complexity'][i],
                scores_dict['forward_iou'][i],
                scores_dict['forward_dice'][i],
                scores_dict['reverse_iou'][i],
                scores_dict['reverse_dice'][i],
                scores_dict['mixed_iou'][i],
                scores_dict['mixed_dice'][i]
            ])
    logging.info(f"Evaluation scores saved to {filename}")

if __name__ == "__main__":
    main() 
