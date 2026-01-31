"""
ASL Model Training Module
=========================
Training loop with early stopping, learning curves, and model export.
"""

import os
import time
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from .model import ASLClassifier, LabelSmoothingCrossEntropy, export_to_onnx
from .dataset import get_dataloaders, ASL_CLASSES


class EarlyStopping:
    """Early stopping to stop training when validation accuracy stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
        elif val_acc < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_acc
            self.counter = 0
        return self.early_stop


class Trainer:
    """
    Trainer class for ASL Classifier.
    
    Args:
        model: ASLClassifier model
        device: Device to train on ('cuda' or 'cpu')
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        
        # Loss function
        self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(val_loader, desc='Validating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        early_stopping_patience: int = 5,
        save_dir: str = 'models'
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save model checkpoints
        
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        best_val_acc = 0.0
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Log history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'asl_cnn_best.pt'))
                print(f"✓ Saved best model (val_acc: {val_acc*100:.2f}%)")
            
            # Early stopping
            if early_stopping(val_acc):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'asl_cnn.pt'))
        print(f"\nTraining complete! Best validation accuracy: {best_val_acc*100:.2f}%")
        
        return self.history
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        train_acc_pct = [a * 100 for a in self.history['train_acc']]
        val_acc_pct = [a * 100 for a in self.history['val_acc']]
        
        axes[1].plot(epochs, train_acc_pct, 'b-', label='Training Accuracy')
        axes[1].plot(epochs, val_acc_pct, 'r-', label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Learning curves saved to {save_path}")
        
        plt.show()


def train_model(
    data_root: str = 'data/asl_alphabet',
    batch_size: int = 64,
    epochs: int = 30,
    learning_rate: float = 1e-4,
    save_dir: str = 'models',
    export_onnx: bool = True
):
    """
    Full training pipeline.
    
    Args:
        data_root: Path to dataset root
        batch_size: Batch size
        epochs: Number of epochs
        learning_rate: Learning rate
        save_dir: Directory to save models
        export_onnx: Whether to export to ONNX after training
    """
    # Get data loaders
    train_loader, val_loader, _ = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        val_split=0.1
    )
    
    # Create model
    model = ASLClassifier(num_classes=len(ASL_CLASSES), pretrained=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=learning_rate
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_dir=save_dir
    )
    
    # Plot learning curves
    trainer.plot_learning_curves(save_path=os.path.join(save_dir, 'learning_curves.png'))
    
    # Export to ONNX
    if export_onnx:
        # Load best model
        model.load_state_dict(torch.load(os.path.join(save_dir, 'asl_cnn_best.pt')))
        export_to_onnx(model, os.path.join(save_dir, 'asl_cnn.onnx'))
    
    return model, history


if __name__ == "__main__":
    train_model()
