"""
ASL Dataset Module
==================
PyTorch Dataset and DataLoaders for ASL Alphabet dataset.
Handles loading, augmentation, and preprocessing of ASL images.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np


# ASL Alphabet classes (29 total: A-Z + SPACE + DELETE + NOTHING)
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'nothing', 'space'
]

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ASLDataset(Dataset):
    """
    PyTorch Dataset for ASL Alphabet images.
    
    Args:
        root_dir: Path to dataset root (e.g., 'data/asl_alphabet/train')
        transform: Optional torchvision transforms to apply
        classes: List of class names (defaults to ASL_CLASSES)
    """
    
    def __init__(
        self, 
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        classes: List[str] = ASL_CLASSES
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_dir}. Check your dataset path.")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Augmentations:
    - Resize to image_size x image_size
    - Random rotation (±15°)
    - Random horizontal flip
    - Color jitter (brightness/contrast)
    - Normalize to ImageNet mean/std
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_dataloaders(
    data_root: str = 'data/asl_alphabet',
    batch_size: int = 64,
    val_split: float = 0.1,
    image_size: int = 224,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        data_root: Root directory containing train/ and optionally test/ folders
        batch_size: Batch size for all loaders
        val_split: Fraction of training data to use for validation
        image_size: Target image size (default 224 for MobileNet/EfficientNet)
        num_workers: Number of data loading workers
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader (test_loader is None if no test set)
    """
    # Handle nested folder structure from Kaggle download
    train_dir = os.path.join(data_root, 'asl_alphabet_train', 'asl_alphabet_train')
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_root, 'asl_alphabet_train')
    
    test_dir = os.path.join(data_root, 'asl_alphabet_test', 'asl_alphabet_test')
    if not os.path.exists(test_dir):
        test_dir = os.path.join(data_root, 'asl_alphabet_test')
    
    # Create datasets
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)
    
    full_train_dataset = ASLDataset(train_dir, transform=train_transform)
    
    # Split into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    torch.manual_seed(seed)
    train_dataset, val_dataset_raw = random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    # Create a validation dataset with val transforms
    # We need to create a wrapper that uses val transforms
    val_dataset = ASLDataset(train_dir, transform=val_transform)
    val_indices = val_dataset_raw.indices
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test loader (if test directory exists with proper structure)
    # Note: Kaggle ASL dataset test folder doesn't have class subfolders
    test_loader = None
    try:
        if os.path.exists(test_dir):
            # Check if test_dir has class subfolders
            test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if test_classes:
                test_dataset = ASLDataset(test_dir, transform=val_transform)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )
    except Exception as e:
        print(f"Note: Test set not loaded ({e})")
    
    print(f"Dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    if test_loader:
        print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Number of classes: {len(ASL_CLASSES)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader, test_loader


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor from ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


if __name__ == "__main__":
    # Test the dataset loading
    print("Testing ASL Dataset loading...")
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_root='data/asl_alphabet',
            batch_size=32,
            val_split=0.1
        )
        
        # Get a sample batch
        images, labels = next(iter(train_loader))
        print(f"\nSample batch shape: {images.shape}")
        print(f"Labels: {[ASL_CLASSES[l] for l in labels[:5]]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the dataset is downloaded to data/asl_alphabet/")
