import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.models.siamese_model import SiameseModel
from data.dataset import WeaklySupervisedMNISTDataset
from src.utils.constants import SIAMESE_TRAINING_DIR

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target.float())
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == target).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({'loss': total_loss / (batch_idx + 1),
                         'acc': 100. * correct / total})
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target.float())
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), 100. * correct / total

def train_siamese_model(
    batch_size=128,
    num_epochs=5,
    learning_rate=0.001,
    weight_decay=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir=SIAMESE_TRAINING_DIR
):
    # Create datasets
    train_dataset = WeaklySupervisedMNISTDataset(split='train', size='50k')
    val_dataset = WeaklySupervisedMNISTDataset(split='val')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SiameseModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_val_acc = 0
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_siamese_model.pth'))
            

if __name__ == '__main__':
    save_dir = f"{SIAMESE_TRAINING_DIR}/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    # Create save directry (including parent directories)
    os.makedirs(save_dir, exist_ok=True)
    train_siamese_model(save_dir=save_dir)
