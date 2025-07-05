#!/bin/bash

#SBATCH -J mae_curriculum_iounonemptypatch   # Job name
#SBATCH -D /dss/dsshome1/02/ge62rob3/mae  # Working directory
#SBATCH -o /dss/dsshome1/02/ge62rob3/mae/mae_curriculum_%j.out # Standard output
#SBATCH -e /dss/dsshome1/02/ge62rob3/mae/mae_curriculum_%j.err # Standard error
#SBATCH --get-user-env       # Get user environment
#SBATCH --export=NONE        # Do not export environment
#SBATCH -p lrz-hgx-h100-94x4 # H100 partition (BayernKI)
#SBATCH --nodes=1            # Number of nodes
#SBATCH --ntasks=1           # Single task
#SBATCH --cpus-per-task=48   # Half of the 96 CPUs per node
#SBATCH --gres=gpu:1         # Request 1 H100 GPU
#SBATCH --mem=192G           # 1/4 of the 768GB node memory
#SBATCH --time=2-00:00:00    # Maximum runtime 2 days

# Initialize conda
source $HOME/miniconda3/etc/profile.d/conda.sh

# Activate our environment
conda activate mae_env3

echo "=== Environment Information ==="
echo "Python version: $(python --version)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Number of CPU cores: $(nproc)"
echo "GPU information:"
nvidia-smi
echo "Available memory:"
free -h
echo "Current directory: $(pwd)"

# Verify PyTorch GPU access and versions
python -c "
import torch
import timm
import tensorboard
from torch.utils.tensorboard import SummaryWriter
print('PyTorch version:', torch.__version__)
print('timm version:', timm.__version__)
print('CUDA available:', torch.cuda.is_available())
print('CUDA device count:', torch.cuda.device_count())
print('CUDA current device:', torch.cuda.current_device())
print('CUDA device name:', torch.cuda.get_device_name(0))
"

# Create output directory
mkdir -p /dss/dsshome1/02/ge62rob3/mae/output_curriculum_cumulative_mae_30epoch_iou_nonemptypatch

# Run the curriculum learning script
python main_curriculum_mae_bw_iou_nonemptypatch.py \
    --epochs_per_stage 30 \
    --batch_size 64 \
    --model mae_vit_base_patch16_bw \
    --mask_ratio 0.25 \
    --data_path /dss/dsshome1/02/ge62rob3/mae/dataset_imagenet_style \
    --output_dir /dss/dsshome1/02/ge62rob3/mae/output_curriculum_cumulative_mae_30epoch_iou_nonemptypatch \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --device cuda \
    --num_workers 48 \
    2>&1 | tee -a /dss/dsshome1/02/ge62rob3/mae/output_curriculum_cumulative_mae_30epoch_iou_nonemptypatch/training_main_curriculum_mae_bw_iou_nonemptypatch.log 
