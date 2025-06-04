import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datetime import datetime

from src.models.siamese_model import SiameseModel
from src.models.weakly_supervised_model import WeaklySupervisedModel
from data.dataset import SupervisedMNISTDataset, WeaklySupervisedMNISTDataset
from src.utils.constants import SIAMESE_TRAINING_DIR, WEAKLY_SUPERVISED_TRAINING_DIR


def save_embeddings(model, model_type, test_loader, epoch, save_dir, device):
    """Save embeddings at specific epochs for visualization"""
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if batch_idx >= 10:  # Save subset for efficiency
                break
            
            images = images.to(device)
            
            if model_type == 'siamese':
                embed = model.extract_embedding(images)
            elif model_type == 'weakly_supervised':
                embed = model.feature_extractor(images)
            
            embeddings.append(embed.cpu().numpy())
            labels.append(targets.numpy())
    
    if embeddings:  # Check if we have any embeddings
        embeddings = np.vstack(embeddings)
        labels = np.hstack(labels)
        
        # Save
        checkpoint_dir = os.path.join(save_dir, 'embeddings')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        np.save(os.path.join(checkpoint_dir, f'embeddings_epoch_{epoch}.npy'), embeddings)
        np.save(os.path.join(checkpoint_dir, f'labels_epoch_{epoch}.npy'), labels)
        print(f"  Saved {len(embeddings)} embeddings for epoch {epoch}")
    
    model.train()


def train_siamese_with_checkpoints(
    batch_size=128,
    num_epochs=5,
    learning_rate=0.001,
    weight_decay=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir=None,
    dataset_size='50k'
):
    """Train siamese model and save embeddings at key epochs"""
    
    if save_dir is None:
        save_dir = f"{SIAMESE_TRAINING_DIR}/{dataset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = WeaklySupervisedMNISTDataset(split='train', size=dataset_size)
    val_dataset = WeaklySupervisedMNISTDataset(split='val')
    test_dataset = SupervisedMNISTDataset(split='test')  # For embedding extraction
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SiameseModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Epochs to save embeddings - adjusted for 5 epoch training
    save_epochs = [1, 2, 3, 5]
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        
        # Train
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
        
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target.float())
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save embeddings at specific epochs
        if epoch in save_epochs:
            save_embeddings(model, 'siamese', test_loader, epoch, save_dir, device)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_siamese_model.pth'))


def train_weakly_supervised_with_checkpoints(
    batch_size=128,
    num_epochs=5,
    learning_rate=0.001,
    weight_decay=1e-4,
    device=None,
    save_dir=None,
    dataset_size='500k'
):
    """Train weakly supervised model and save embeddings at key epochs"""
    
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    
    if save_dir is None:
        save_dir = f"{WEAKLY_SUPERVISED_TRAINING_DIR}/{dataset_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset = WeaklySupervisedMNISTDataset(split='train', size=dataset_size)
    val_dataset = WeaklySupervisedMNISTDataset(split='val')
    test_dataset = SupervisedMNISTDataset(split='test')  # For embedding extraction
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize model
    model = WeaklySupervisedModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Epochs to save embeddings - adjusted for 5 epoch training
    save_epochs = [1, 2, 3, 5]
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            outputs = torch.clamp(outputs, 0, 1)
            loss = criterion(outputs, target.float())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'loss': total_loss / (batch_idx + 1),
                             'acc': 100. * correct / total})
        
        train_acc = 100. * correct / total
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                outputs = torch.clamp(outputs, 0, 1)
                loss = criterion(outputs, target.float())
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == target).sum().item()
                val_total += target.size(0)
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save embeddings at specific epochs
        if epoch in save_epochs:
            save_embeddings(model, 'weakly_supervised', test_loader, epoch, save_dir, device)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(save_dir, 'best_weakly_supervised_model.pth'))


# Also need to train supervised model with checkpoints
def train_supervised_with_checkpoints(
    batch_size=64,
    num_epochs=5,
    learning_rate=0.001,
    weight_decay=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir=None
):
    """Train supervised model and save embeddings at key epochs"""
    
    from src.models.supervised_model import SupervisedModel
    from src.utils.constants import SUPERVISED_TRAINING_DIR
    
    if save_dir is None:
        save_dir = f"{SUPERVISED_TRAINING_DIR}/checkpoints_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = SupervisedMNISTDataset(split='train')
    val_dataset = SupervisedMNISTDataset(split='val')
    test_dataset = SupervisedMNISTDataset(split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Initialize model
    model = SupervisedModel().to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Epochs to save embeddings
    save_epochs = [1, 2, 3, 5]
    
    # Training loop
    best_val_acc = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': total_loss/(batch_idx+1), 'acc': 100.*correct/total})
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save embeddings at specific epochs
        if epoch in save_epochs:
            model.eval()
            embeddings = []
            labels_list = []
            
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    if batch_idx >= 10:
                        break
                    images = images.to(device)
                    embed = model.get_embedding(images)
                    embeddings.append(embed.cpu().numpy())
                    labels_list.append(labels.numpy())
            
            embeddings = np.vstack(embeddings)
            labels_array = np.hstack(labels_list)
            
            checkpoint_dir = os.path.join(save_dir, 'embeddings')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            np.save(os.path.join(checkpoint_dir, f'embeddings_epoch_{epoch}.npy'), embeddings)
            np.save(os.path.join(checkpoint_dir, f'labels_epoch_{epoch}.npy'), labels_array)
            print(f"  Saved {len(embeddings)} embeddings for epoch {epoch}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))


if __name__ == '__main__':
    # Train all models with checkpoints
    print("Training Supervised CNN...")
    train_supervised_with_checkpoints()
    
    print("\n\nTraining Siamese model with 50k pairs...")
    train_siamese_with_checkpoints(dataset_size='50k', num_epochs=5)
    
    print("\n\nTraining Weakly Supervised model with 500k pairs...")
    train_weakly_supervised_with_checkpoints(dataset_size='500k', num_epochs=5)