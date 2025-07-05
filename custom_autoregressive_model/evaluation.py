import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE, DATASET_PATH, PATCH_SIZE, MASK_RATIO
from dataset import TangramDataset, transform

def compute_iou_and_dice(pred_bin, gt_bin):
    """
    Compute IoU and Dice for a single mask vs. ground truth (both 0..1).
    pred_bin, gt_bin: shape [patch_size^2].
    """
    intersection = (pred_bin * gt_bin).sum()
    union = (pred_bin + gt_bin - pred_bin * gt_bin).sum()
    iou = intersection / (union + 1e-8)

    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()
    dice = 2.0 * intersection / (pred_sum + gt_sum + 1e-8)
    return iou.item(), dice.item()

def evaluate_model_on_complexity(model, complexity_folder):
    """
    Evaluate the model on all images in a specific tangram complexity folder
    (e.g., tangrams_3_piece). Returns average IoU and Dice across the dataset.
    """
    model.eval()
    dataset = TangramDataset(complexity_folder, patch_size=PATCH_SIZE, mask_ratio=MASK_RATIO, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    iou_list, dice_list = [], []
    with torch.no_grad():
        for input_patches, mask, target_patches in dataloader:
            input_patches = input_patches.to(DEVICE)
            mask = mask.to(DEVICE)
            target_patches = target_patches.to(DEVICE)

            logits = model(input_patches, mask)
            probs = torch.sigmoid(logits)
            pred_bin = (probs >= 0.5).float()
            gt_bin = (target_patches >= 0.5).float()

            # Compute IoU, Dice for these masked patches
            iou, dice = compute_iou_and_dice(pred_bin, gt_bin)
            iou_list.append(iou)
            dice_list.append(dice)

    avg_iou = np.mean(iou_list) if iou_list else 0
    avg_dice = np.mean(dice_list) if dice_list else 0
    return avg_iou, avg_dice 