import torch

# -----------------
# HYPERPARAMETERS
# -----------------
IMAGE_SIZE = 256
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 256/16=16 => 16*16=256 patches
EMBED_DIM = PATCH_SIZE * PATCH_SIZE  # Single channel => 16x16=256
MASK_RATIO = 0.15
NUM_HEADS = 8
NUM_LAYERS = 6
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 40          # Adjust as desired for each stage or for the mixed dataset
PATIENCE = 50         # Early stopping patience

# Use MPS (Metal Performance Shaders) for M-series Macs if available
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Use your local dataset path - adjust this to your actual path
DATASET_PATH = "dataset"  # Default local path

# Forward order subfolders
FORWARD_FOLDERS = [
    "tangrams_1_piece",
    "tangrams_2_piece",
    "tangrams_3_piece",
    "tangrams_4_piece",
    "tangrams_5_piece",
    "tangrams_6_piece",
    "tangrams_7_piece",
]

# Reverse order subfolders
REVERSE_FOLDERS = list(reversed(FORWARD_FOLDERS))

# Checkpoint saving frequency
# CHECKPOINT_FREQUENCY = 10 