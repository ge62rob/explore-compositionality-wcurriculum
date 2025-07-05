# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

from timm.data import create_transform


GRAYSCALE_MEAN = [0.1163143590092659, 0.1163143590092659, 0.1163143590092659]
GRAYSCALE_STD = [0.31751421093940735, 0.31751421093940735, 0.31751421093940735]


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = TangramDatasetBW(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = GRAYSCALE_MEAN
    std = GRAYSCALE_STD
    # train transform
    if is_train:
        # Create augmentation transform for grayscale
        t = []
        t.append(transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=PIL.Image.BICUBIC))
        t.append(transforms.RandomHorizontalFlip())
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class TangramDatasetBW(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # Look for images in all subdirectories
        print(f"Looking for images in: {root_dir}")
        
        # Check if directory exists
        if not os.path.exists(root_dir):
            print(f"WARNING: Directory not found at {root_dir}")
            return
            
        # Find all image files recursively in all class subdirectories
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(subdir, file))
        
        print(f"Found {len(self.image_paths)} images")
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image as grayscale (single channel)
        image = Image.open(img_path).convert('L')
            
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Apply thresholding directly here
        image = (image > 0.5).float()   
        # For training, we don't need labels
        return image, 0  # 0 is a dummy label 
