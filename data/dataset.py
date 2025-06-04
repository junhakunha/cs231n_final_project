import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import random

from src.utils.constants import HOME_DIR, DATA_DIR, MNIST_DIR, SUPERVISED_DIR, WEAKLY_SUPERVISED_DIR


class SupervisedMNISTDataset(Dataset):
    def __init__(self, split='train', transform=None):
        """
        Initialize the MNIST dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.split = split
        self.transform = transform
        
        # Load the appropriate data split
        if split == 'train':
            self.data = torch.load(f"{SUPERVISED_DIR}/x_train.pt")
            self.labels = torch.load(f"{SUPERVISED_DIR}/y_train.pt")
        elif split == 'val':
            self.data = torch.load(f"{SUPERVISED_DIR}/x_val.pt")
            self.labels = torch.load(f"{SUPERVISED_DIR}/y_val.pt")
        elif split == 'test':
            self.data = torch.load(f"{SUPERVISED_DIR}/x_test.pt")
            self.labels = torch.load(f"{SUPERVISED_DIR}/y_test.pt")
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
        
        # Convert data to float and normalize to [0, 1]
        self.data = self.data.float() / 255.0
        
        # Add channel dimension if not present
        if len(self.data.shape) == 3:
            self.data = self.data.unsqueeze(1)
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    

class WeaklySupervisedMNISTDataset(Dataset):
    def __init__(self, split='train', size='50k', transform=None):
        """
        Initialize the weakly supervised MNIST dataset.
        
        Args:
            split (str): One of 'train', 'val', or 'test'
            size (str): For training split, one of '50k', '500k', or '5000k'. Ignored for val/test.
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.split = split
        self.size = size
        self.transform = transform
        
        # Load the appropriate data split
        if split == 'train':
            if size not in ['50k', '500k', '5000k']:
                raise ValueError(f"Invalid size: {size}. Must be one of '50k', '500k', or '5000k'")
            self.data = torch.load(f"{WEAKLY_SUPERVISED_DIR}/x_train_{size}.pt")
            self.labels = torch.load(f"{WEAKLY_SUPERVISED_DIR}/y_train_{size}.pt")
        elif split == 'val':
            self.data = torch.load(f"{WEAKLY_SUPERVISED_DIR}/x_val.pt")
            self.labels = torch.load(f"{WEAKLY_SUPERVISED_DIR}/y_val.pt")
        elif split == 'test':
            self.data = torch.load(f"{WEAKLY_SUPERVISED_DIR}/x_test.pt")
            self.labels = torch.load(f"{WEAKLY_SUPERVISED_DIR}/y_test.pt")
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
        
        # Convert data to float and normalize to [0, 1]
        self.data = self.data.float() / 255.0
        
        # Add channel dimension if not present
        if len(self.data.shape) == 4:  # (N, 2, H, W)
            self.data = self.data.unsqueeze(2)  # (N, 2, 1, H, W)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        img_pair = self.data[idx]  # Shape: (2, 1, H, W)
        label = self.labels[idx]    # Shape: (1,)
        
        if self.transform:
            img_pair = self.transform(img_pair)
            
        return img_pair, label