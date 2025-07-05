import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import config

from config import DEVICE, EPOCHS, PATIENCE, BATCH_SIZE, LEARNING_RATE
from config import DATASET_PATH, FORWARD_FOLDERS, REVERSE_FOLDERS
from config import PATCH_SIZE, MASK_RATIO, NUM_PATCHES, EMBED_DIM, NUM_HEADS, NUM_LAYERS
from dataset import TangramDataset, MergedTangramDataset, transform
from model import AutoregressiveTransformer

def train_on_dataset(model, dataloader, optimizer, scheduler, criterion, epochs, patience, device,
                     curriculum_stage=None, output_dir='', stage_idx=None):
    """
    Train the model on a single dataset (e.g., 1-piece tangrams).
    Returns:
      model (updated),
      losses (list of average losses per epoch for plotting).
    """
    model.train()
    best_loss = float('inf')
    early_stop_counter = 0
    epoch_losses = []
    best_model_state = None

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_idx, (input_patches, mask, target_patches) in enumerate(dataloader):
            input_patches = input_patches.to(device)
            mask = mask.to(device)
            target_patches = target_patches.to(device)

            optimizer.zero_grad()
            logits = model(input_patches, mask)

            # Ensure target_patches shape matches logits
            if logits.shape != target_patches.shape:
                target_patches = target_patches.view_as(logits)

            loss = criterion(logits, target_patches)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        if curriculum_stage is not None:
            print(f"[{curriculum_stage}] Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")

        # Step the LR scheduler
        scheduler.step(avg_loss)

        # Track best model but don't save yet
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            # Save the best model state - we'll return this instead of saving files for each stage
            best_model_state = model.state_dict().copy()
            print(f"New best model at epoch {epoch} with loss {avg_loss:.4f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered (no improvement for {patience} epochs).")
                break

    # Load the best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model, epoch_losses

def train_forward_curriculum(output_dir=''):
    """
    Trains a new model from scratch in ascending order of tangram complexity:
    1 -> 2 -> 3 -> ... -> 7 pieces.
    """
    print("=== TRAINING: FORWARD CURRICULUM (1 -> 7) ===")
    # Initialize model & optimizer
    model = AutoregressiveTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_patches=NUM_PATCHES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    all_epoch_losses = []
    
    # Train stage by stage
    for stage_idx, stage_folder in enumerate(FORWARD_FOLDERS):
        folder_path = os.path.join(DATASET_PATH, stage_folder)
        dataset = TangramDataset(folder_path, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        
        print(f"[TangramDataset] {folder_path} -> Total images found: {len(dataset)}")

        # Train on this complexity level
        model, stage_losses = train_on_dataset(
            model, dataloader, optimizer, scheduler, criterion,
            epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
            curriculum_stage=stage_folder,
            stage_idx=stage_idx
        )
        all_epoch_losses.extend(stage_losses)

    # Save only one final model at the end of all training
    final_model_path = os.path.join(output_dir, "best_autoregressive_transformer_forward.pth") if output_dir else "best_autoregressive_transformer_forward.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final forward curriculum model to {final_model_path}")
    
    return model, all_epoch_losses

def train_reverse_curriculum(output_dir=''):
    """
    Trains a new model from scratch in descending order of tangram complexity:
    7 -> 6 -> 5 -> ... -> 1 pieces.
    """
    print("=== TRAINING: REVERSE CURRICULUM (7 -> 1) ===")
    model = AutoregressiveTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_patches=NUM_PATCHES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    all_epoch_losses = []
    for stage_folder in REVERSE_FOLDERS:
        folder_path = os.path.join(DATASET_PATH, stage_folder)
        dataset = TangramDataset(folder_path, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        model, stage_losses = train_on_dataset(
            model, dataloader, optimizer, scheduler, criterion,
            epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
            curriculum_stage=stage_folder,
            output_dir=output_dir
        )
        all_epoch_losses.extend(stage_losses)

    # Save the final model
    final_model_path = os.path.join(output_dir, "best_autoregressive_transformer_reverse.pth") if output_dir else "best_autoregressive_transformer_reverse.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return model, all_epoch_losses

def train_mixed_curriculum(output_dir=''):
    """
    Trains a new model from scratch using a single merged dataset that includes
    all tangram complexities (1 to 7 pieces) in random order.
    """
    print("=== TRAINING: MIXED CURRICULUM (all pieces together) ===")
    # Create a merged dataset from all subfolders
    # One way: gather all images from 1_piece to 7_piece in a single list
    all_image_paths = []
    for folder in FORWARD_FOLDERS:
        subdir_path = os.path.join(DATASET_PATH, folder)
        for file in os.listdir(subdir_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(subdir_path, file))

    mixed_dataset = MergedTangramDataset(
        all_image_paths, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform
    )
    print(f"[MixedTangramDataset] Total images found: {len(mixed_dataset)}")
    mixed_dataloader = DataLoader(mixed_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Initialize model & optimizer
    model = AutoregressiveTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_patches=NUM_PATCHES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    model, epoch_losses = train_on_dataset(
        model, mixed_dataloader, optimizer, scheduler, criterion,
        epochs=EPOCHS, patience=PATIENCE, device=DEVICE,
        curriculum_stage="Mixed",
        output_dir=output_dir
    )

    # Save the final model
    final_model_path = os.path.join(output_dir, "best_autoregressive_transformer_mixed.pth") if output_dir else "best_autoregressive_transformer_mixed.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    return model, epoch_losses 

def train_forward_curriculum_singlerun(output_dir='results_singlerun', epochs=100):
    """
    Trains a model in a single run with forward curriculum order (1->7 pieces),
    ensuring data is presented in ascending complexity order.
    """
    print("=== TRAINING: FORWARD CURRICULUM (SINGLE RUN) ===")
    
    # Initialize model & optimizer
    model = AutoregressiveTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_patches=NUM_PATCHES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gather all images in complexity order (1->7)
    all_image_paths = []
    for folder in FORWARD_FOLDERS:
        subdir_path = os.path.join(DATASET_PATH, folder)
        folder_images = []
        
        if os.path.exists(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    folder_images.append(os.path.join(subdir_path, file))
            
            # Add all images from this complexity level 
            all_image_paths.extend(folder_images)
            print(f"Added {len(folder_images)} images from {folder}")
    
    # Create dataset and dataloader
    dataset = MergedTangramDataset(
        all_image_paths, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform
    )
    print(f"[ForwardOrderedDataset] Total images: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Train in a single run
    model, epoch_losses = train_on_dataset(
        model, dataloader, optimizer, scheduler, criterion,
        epochs=epochs, patience=epochs//5, # Set patience as 20% of epochs 
        device=DEVICE,
        curriculum_stage="Forward-SingleRun",
        output_dir=output_dir
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "best_autoregressive_transformer_forward_singlerun.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final forward curriculum model to {final_model_path}")
    
    return model, epoch_losses

def train_reverse_curriculum_singlerun(output_dir='results_singlerun', epochs=100):
    """
    Trains a model in a single run with reverse curriculum order (7->1 pieces),
    ensuring data is presented in descending complexity order.
    """
    print("=== TRAINING: REVERSE CURRICULUM (SINGLE RUN) ===")
    
    # Initialize model & optimizer
    model = AutoregressiveTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_patches=NUM_PATCHES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gather all images in reverse complexity order (7->1)
    all_image_paths = []
    for folder in REVERSE_FOLDERS:
        subdir_path = os.path.join(DATASET_PATH, folder)
        folder_images = []
        
        if os.path.exists(subdir_path):
            for file in os.listdir(subdir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    folder_images.append(os.path.join(subdir_path, file))
            
            # Add all images from this complexity level
            all_image_paths.extend(folder_images)
            print(f"Added {len(folder_images)} images from {folder}")
    
    # Create dataset and dataloader
    dataset = MergedTangramDataset(
        all_image_paths, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform
    )
    print(f"[ReverseOrderedDataset] Total images: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Train in a single run
    model, epoch_losses = train_on_dataset(
        model, dataloader, optimizer, scheduler, criterion,
        epochs=epochs, patience=epochs//5, # Set patience as 20% of epochs
        device=DEVICE,
        curriculum_stage="Reverse-SingleRun",
        output_dir=output_dir
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "best_autoregressive_transformer_reverse_singlerun.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final reverse curriculum model to {final_model_path}")
    
    return model, epoch_losses

def train_mixed_curriculum_singlerun(output_dir='results_singlerun', epochs=100):
    """
    Trains a model in a single run with mixed curriculum (random order),
    which is the same as the original mixed curriculum but with explicit epochs.
    """
    print("=== TRAINING: MIXED CURRICULUM (SINGLE RUN) ===")
    
    # Initialize model & optimizer
    model = AutoregressiveTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        num_patches=NUM_PATCHES
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gather all images (same as original mixed)
    all_image_paths = []
    for folder in FORWARD_FOLDERS:
        subdir_path = os.path.join(DATASET_PATH, folder)
        for file in os.listdir(subdir_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(subdir_path, file))
    
    # Create dataset and dataloader (with shuffle=True for mixed curriculum)
    dataset = MergedTangramDataset(
        all_image_paths, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform
    )
    print(f"[MixedDataset] Total images: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Train in a single run
    model, epoch_losses = train_on_dataset(
        model, dataloader, optimizer, scheduler, criterion,
        epochs=epochs, patience=epochs//5, # Set patience as 20% of epochs
        device=DEVICE,
        curriculum_stage="Mixed-SingleRun",
        output_dir=output_dir
    )
    
    # Save the final model
    final_model_path = os.path.join(output_dir, "best_autoregressive_transformer_mixed_singlerun.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final mixed curriculum model to {final_model_path}")
    
    return model, epoch_losses
