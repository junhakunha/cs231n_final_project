import sys
import os
import torch
from torchvision import datasets
import numpy as np

from src.utils.constants import HOME_DIR, DATA_DIR, MNIST_DIR, SUPERVISED_DIR, WEAKLY_SUPERVISED_DIR


def download_mnist():
    # Create MNIST directory if it doesn't exist
    os.makedirs(MNIST_DIR, exist_ok=True)
    os.makedirs(SUPERVISED_DIR, exist_ok=True)
    os.makedirs(WEAKLY_SUPERVISED_DIR, exist_ok=True)
    
    # Download training data
    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=None
    )
    
    # Download test data
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=None
    )

    # Convert PIL images to tensors
    train_data = torch.stack([torch.tensor(np.array(img)) for img, _ in train_dataset])
    train_labels = torch.tensor([label for _, label in train_dataset])
    
    test_data = torch.stack([torch.tensor(np.array(img)) for img, _ in test_dataset])
    test_labels = torch.tensor([label for _, label in test_dataset])

    # Split training data into train and validation sets
    train_size = int(5/6 * len(train_data))
    val_size = len(train_data) - train_size
    
    train_data, val_data = torch.split(train_data, [train_size, val_size])
    train_labels, val_labels = torch.split(train_labels, [train_size, val_size])

    # Save the datasets as tensors: x_train, y_train, x_val, y_val, x_test, y_test
    torch.save(train_data, f"{SUPERVISED_DIR}/x_train.pt")
    torch.save(train_labels, f"{SUPERVISED_DIR}/y_train.pt")
    torch.save(val_data, f"{SUPERVISED_DIR}/x_val.pt")
    torch.save(val_labels, f"{SUPERVISED_DIR}/y_val.pt")
    torch.save(test_data, f"{SUPERVISED_DIR}/x_test.pt")
    torch.save(test_labels, f"{SUPERVISED_DIR}/y_test.pt")


def construct_weakly_supervised_dataset(seed=42):
    """
    Construct a weakly supervised dataset from the MNIST dataset.

    The weakly supervised dataset is a relational dataset where each training example is a pair of images, and the
    label is 0 if the first image is smaller than the second image, 1 if the first image is larger than the second image.
    Images with the same label are NOT included in the dataset.

    The dataset is saved as a tensor of shape (N, 2, 28, 28), where N is the number of training/validation/test examples.
    The label is saved as a tensor of shape (N, 1)

    We construct 3 different training sets, by size. They are constructed by randomly sampling the pairs from the supervised training set.
    - 50000 image pairs
    - 500000 image pairs
    - 5000000 image pairs

    The validation set is constructed by randomly sampling 20000 image pairs from the supervised validation set.

    The test set is constructed by randomly sampling 20000 image pairs from the supervised test set.
    """
    print("Constructing weakly supervised dataset...")

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    supervised_x_train = torch.load(f"{SUPERVISED_DIR}/x_train.pt")
    supervised_y_train = torch.load(f"{SUPERVISED_DIR}/y_train.pt")
    supervised_x_val = torch.load(f"{SUPERVISED_DIR}/x_val.pt")
    supervised_y_val = torch.load(f"{SUPERVISED_DIR}/y_val.pt")
    supervised_x_test = torch.load(f"{SUPERVISED_DIR}/x_test.pt")
    supervised_y_test = torch.load(f"{SUPERVISED_DIR}/y_test.pt")

    def create_pairs(x_data, y_data, num_pairs):
        # Initialize tensors to store pairs and labels
        x_pairs = torch.zeros((num_pairs, 2, 28, 28), dtype=x_data.dtype)
        y_pairs = torch.zeros((num_pairs, 1), dtype=torch.float)
        
        # Keep track of how many valid pairs we've created
        pairs_created = 0
        
        while pairs_created < num_pairs:
            # Randomly sample two indices
            idx1, idx2 = torch.randint(0, len(x_data), (2,))
            
            # Skip if the digits are the same
            if y_data[idx1] == y_data[idx2]:
                continue
                
            # Add the pair and its label
            x_pairs[pairs_created, 0] = x_data[idx1]
            x_pairs[pairs_created, 1] = x_data[idx2]
            y_pairs[pairs_created] = (y_data[idx1] > y_data[idx2]).float()
            
            pairs_created += 1
            
        return x_pairs, y_pairs

    # Create training sets of different sizes
    train_sizes = [50000, 500000, 5000000]
    for size in train_sizes:
        print(f"Creating training set of size {size}...")
        x_train_pairs, y_train_pairs = create_pairs(supervised_x_train, supervised_y_train, size)
        torch.save(x_train_pairs, f"{WEAKLY_SUPERVISED_DIR}/x_train_{size//1000}k.pt")
        torch.save(y_train_pairs, f"{WEAKLY_SUPERVISED_DIR}/y_train_{size//1000}k.pt")

    # Create validation set
    x_val_pairs, y_val_pairs = create_pairs(supervised_x_val, supervised_y_val, 20000)
    torch.save(x_val_pairs, f"{WEAKLY_SUPERVISED_DIR}/x_val.pt")
    torch.save(y_val_pairs, f"{WEAKLY_SUPERVISED_DIR}/y_val.pt")

    # Create test set
    x_test_pairs, y_test_pairs = create_pairs(supervised_x_test, supervised_y_test, 20000)
    torch.save(x_test_pairs, f"{WEAKLY_SUPERVISED_DIR}/x_test.pt")
    torch.save(y_test_pairs, f"{WEAKLY_SUPERVISED_DIR}/y_test.pt")





if __name__ == "__main__":
    download_mnist()
    construct_weakly_supervised_dataset()

