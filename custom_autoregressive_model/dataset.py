import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES

# Image transform: no extra normalization—just convert to 0..1 range.
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),  # single-channel -> shape [1,256,256], range [0,1]
])

class TangramDataset(Dataset):
    """
    Expects a folder structure like:
      dataset/
        tangrams_1_piece/
          img1.png
          img2.png
        tangrams_2_piece/
          ...
    or a single subfolder with images, depending on root_dir usage.
    """
    def __init__(self, root_dir, patch_size=16, mask_ratio=0.15, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.transform = transform

        # Collect all image file paths
        self.image_paths = []
        # If root_dir is a single folder (e.g., dataset/tangrams_1_piece)
        if os.path.isdir(root_dir):
            for file in os.listdir(root_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root_dir, file))
        # Or if root_dir is the main dataset folder containing multiple subfolders
        else:
            raise ValueError("root_dir must be a valid directory path")

        print(f"[TangramDataset] {root_dir} -> Total images found: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)  # shape [1,256,256], range [0,1]

        # Split into patches => shape [num_patches, patch_size*patch_size]
        patches = self.image_to_patches(image, self.patch_size)  # [256, 256]

        # Create masking indices
        num_mask = int(self.mask_ratio * NUM_PATCHES)
        mask_indices = random.sample(range(NUM_PATCHES), num_mask)
        mask = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        mask[mask_indices] = True

        # Prepare input and target
        input_patches = patches.clone()
        input_patches[mask] = 0.0  # fill masked patches with zeros

        target_patches = patches[mask]

        return input_patches, mask, target_patches

    def image_to_patches(self, image, patch_size):
        """
        Splits the image tensor [1,H,W] into (H/patch_size)*(W/patch_size) patches,
        each flattened to [patch_size*patch_size].
        """
        C, H, W = image.shape
        assert H == IMAGE_SIZE and W == IMAGE_SIZE, "Image must be resized to 256x256"

        # [1,H,W] -> unfold over dim=1, then dim=2
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        # now reshape to [C, num_patches, patch_size, patch_size]
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        # flatten each patch => [num_patches, patch_size*patch_size]
        patches = patches.squeeze(0).reshape(patches.size(1), -1)
        return patches

class MergedTangramDataset(Dataset):
    def __init__(self, image_paths, patch_size=16, mask_ratio=0.15, transform=None):
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        patches = self.image_to_patches(image, self.patch_size)
        num_mask = int(self.mask_ratio * NUM_PATCHES)
        mask_indices = random.sample(range(NUM_PATCHES), num_mask)
        mask = torch.zeros(NUM_PATCHES, dtype=torch.bool)
        mask[mask_indices] = True
        input_patches = patches.clone()
        input_patches[mask] = 0.0
        target_patches = patches[mask]
        return input_patches, mask, target_patches

    def image_to_patches(self, image, patch_size):
        C, H, W = image.shape
        patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(C, -1, patch_size, patch_size)
        patches = patches.squeeze(0).reshape(patches.size(1), -1)
        return patches 