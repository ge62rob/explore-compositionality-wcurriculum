import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
import random
import argparse

# Add the mae directory to the Python path so models_mae_bw can be found
mae_dir = os.path.join(os.getcwd(), 'mae')
sys.path.append(mae_dir)
import models_mae_bw


def show_image(image, title=''):
    # For binary images, just show directly
    # If image has shape [C, H, W] or [1, H, W], squeeze it to [H, W]
    if isinstance(image, torch.Tensor):
        image = image.squeeze()
    elif isinstance(image, np.ndarray):
        image = np.squeeze(image)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

'''
def show_image(image, title=''):
    # Ensure that if image is a torch.Tensor, we detach, move it to CPU, and convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().squeeze().numpy()
    elif isinstance(image, np.ndarray):
        image = np.squeeze(image)
    plt.imshow(image, cmap='gray', vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return
'''
def prepare_model(chkpt_dir, arch='mae_vit_base_patch16_bw'):
    # build model
    model = getattr(models_mae_bw, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    # If checkpoint is already a state dict, use it directly
    state_dict = checkpoint if not isinstance(checkpoint, dict) else checkpoint.get('model', checkpoint)
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)
    return model

def run_one_image(img, model, mask_ratio=0.75):
    # img is already in [C, H, W] format
    x = img.clone()
    
    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = x.float()

    # run MAE
    loss, y, mask = model(x, mask_ratio=mask_ratio)
    y = model.unpatchify(y)
    
    # Apply sigmoid and threshold for binary predictions
    y = torch.sigmoid(y)
    y = (y > 0.5).float()  # Convert to binary 0/1
    
    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2)
    mask = model.unpatchify(mask)
    
    x = x.detach().cpu()
    y = y.detach().cpu()
    mask = mask.detach().cpu()

    # Make reconstruction with visible patches
    y_with_vis = y.clone()
    y_with_vis[~mask.bool()] = x[~mask.bool()]

    return x[0], y[0], mask[0], y_with_vis[0]

'''
def run_one_image(img, model, mask_ratio=0.75):
    # img is assumed to be in [C, H, W] format
    x = img.clone()
    
    # Make it a batch (add batch dimension)
    x = x.unsqueeze(dim=0)
    x = x.float()

    # Run the MAE model
    loss, y, mask = model(x, mask_ratio=mask_ratio)
    # --- Key change: apply sigmoid so predictions are in [0,1] ---
    y = torch.sigmoid(y)
    # Unpatchify the predictions to reconstruct the full image
    y = model.unpatchify(y)
    
    # Visualize the mask:
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2)
    mask = model.unpatchify(mask)
    
    # Detach and move all tensors to CPU
    x = x.detach().cpu()
    y = y.detach().cpu()
    mask = mask.detach().cpu()
    
    # Make reconstruction with visible patches
    y_with_vis = y.clone()
    y_with_vis[~mask.bool()] = x[~mask.bool()]

    return x[0], y[0], mask[0], y_with_vis[0]
'''
#/scratch_models/scratch_model_complexity_1.pth
#default='./output_tangram_bw_mse_hybrid_weightedbce_globalpos_mask025/checkpoint-40.pth'
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize MAE BW reconstruction')
    parser.add_argument('--checkpoint', type=str, default='./scratch_models/scratch_model_complexity_1.pth',
                      help='path to the model checkpoint')
    parser.add_argument('--image', type=str, default='./dataset_imagenet_style/val/tangrams_1_piece/1piece_0900_normal.png',
                      help='path to the image file')
    parser.add_argument('--mask_ratio', type=float, default=0.25,
                      help='masking ratio for visualization (default: 0.75)')
    
    args = parser.parse_args()
    
    # Load the model
    chkpt_path = args.checkpoint
    print(f"Loading model from: {chkpt_path}")
    model = prepare_model(chkpt_path)
    model.global_pos_weight = torch.tensor([7.593], device='cpu')
    model.eval()
    
    # Use the specified image path
    img_path = args.image
    print(f"Using image: {img_path}")
    
    # Ensure the image file exists
    if not os.path.exists(img_path):
        print(f"ERROR: Image file not found: {img_path}")
        # If image not found, try to find an alternative
        print("Looking for alternative images...")
        val_dir = os.path.dirname(os.path.dirname(img_path))
        found = False
        for root, dirs, files in os.walk(val_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(root, file)
                    print(f"Found alternative image: {img_path}")
                    found = True
                    break
            if found:
                break
        
        if not found:
            print("No image files found. Exiting.")
            sys.exit(1)
    
    # Load and preprocess the image
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # Convert to binary - threshold at 0.5 since ToTensor scales to [0,1]
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    img_tensor = transform(img)
    # Keep the tensor in [C, H, W] format - no need to permute
    img_for_model = img_tensor
    
    # Run reconstruction and visualization
    print(f"Running MAE BW reconstruction with mask_ratio={args.mask_ratio}...")
    x_orig, y_pred, mask, y_with_vis = run_one_image(img_for_model, model, args.mask_ratio)
    print("Done! Image saved as 'bw_reconstruction.png'")

    # visualization
    print('saving visualization to bw_reconstruction.png')
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    show_image(x_orig, "Original")

    plt.subplot(1, 4, 2)
    im_masked = x_orig * (1 - mask)
    show_image(im_masked, f"Masked")
    '''
    plt.subplot(1, 4, 3)
    show_image(y_pred, "Reconstruction")
    '''
    # 3) Only the model's predictions on masked patches
    recon_mask_only = y_pred.clone()
    recon_mask_only[~mask.bool()] = 0
    plt.subplot(1, 4, 3)
    show_image(recon_mask_only, "Reconstruction\n(masked only)")


    plt.subplot(1, 4, 4)
    show_image(y_with_vis, "Reconstruction + Visible")

    plt.savefig('bw_reconstruction.png')
    plt.close() 