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
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from .model import ASLClassifier, LabelSmoothingCrossEntropy, export_to_onnx
from .dataset import get_dataloaders, ASL_CLASSES


class EarlyStopping:
    """Early stopping to stop training when validation accuracy stops improving or drops after high accuracy."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001, high_accuracy_threshold: float = 0.9998):
        self.patience = patience
        self.min_delta = min_delta
        self.high_accuracy_threshold = high_accuracy_threshold
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.reached_high_accuracy = False
    
    def __call__(self, val_acc: float) -> bool:
        if self.best_score is None:
            self.best_score = val_acc
        else:
            # Check if we've reached high accuracy threshold
            if val_acc >= self.high_accuracy_threshold:
                self.reached_high_accuracy = True
            
            # If we reached high accuracy and now it drops, stop immediately
            if self.reached_high_accuracy and val_acc < self.best_score:
                print(f"\n⚠️  Accuracy dropped from {self.best_score*100:.4f}% to {val_acc*100:.4f}% after reaching high accuracy threshold")
                self.early_stop = True
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
        save_dir: str = 'models',
        high_accuracy_threshold: float = 0.9998
    ) -> Dict[str, List[float]]:
        """
        Full training loop with enhanced early stopping.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_dir: Directory to save model checkpoints
            high_accuracy_threshold: Stop if accuracy drops after reaching this threshold
        
        Returns:
            Training history dictionary
        """
        os.makedirs(save_dir, exist_ok=True)
        early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            high_accuracy_threshold=high_accuracy_threshold
        )
        best_val_acc = 0.0
        best_epoch = 0
        
        print(f"Training on {self.device}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"High accuracy threshold for early stopping: {high_accuracy_threshold*100:.2f}%")
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
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.4f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.4f}%")
            
            # Save model every epoch
            epoch_model_path = os.path.join(save_dir, f'asl_cnn_epoch_{epoch+1}.pt')
            torch.save(self.model.state_dict(), epoch_model_path)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), os.path.join(save_dir, 'asl_cnn_best.pt'))
                print(f"✓ Saved best model (val_acc: {val_acc*100:.4f}%)")
            
            # Early stopping
            if early_stopping(val_acc):
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                print(f"📊 Best model from epoch {best_epoch} with accuracy: {best_val_acc*100:.4f}%")
                break
        
        # Save final model
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'asl_cnn_final.pt'))
        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc*100:.4f}% at epoch {best_epoch}")
        
        return self.history
    
    def plot_learning_curves(self, save_path: Optional[str] = None):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.history['train_loss'], 'b-', marker='o', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', marker='s', label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        train_acc_pct = [a * 100 for a in self.history['train_acc']]
        val_acc_pct = [a * 100 for a in self.history['val_acc']]
        
        axes[1].plot(epochs, train_acc_pct, 'b-', marker='o', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, val_acc_pct, 'r-', marker='s', label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Add max accuracy annotation
        max_val_acc = max(val_acc_pct)
        max_epoch = val_acc_pct.index(max_val_acc) + 1
        axes[1].annotate(f'Max: {max_val_acc:.4f}%\nEpoch {max_epoch}',
                        xy=(max_epoch, max_val_acc),
                        xytext=(max_epoch + 1, max_val_acc - 2),
                        fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Learning curves saved to {save_path}")
        
        plt.show()
    
    @torch.no_grad()
    def evaluate_with_confusion_matrix(self, data_loader: DataLoader, class_names: List[str]) -> Tuple[np.ndarray, Dict[str, float]]:
        """Evaluate model and return predictions for confusion matrix."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Compute per-class accuracy
        per_class_acc = {}
        for i, class_name in enumerate(class_names):
            class_mask = all_labels == i
            if class_mask.sum() > 0:
                class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
                per_class_acc[class_name] = class_correct / class_mask.sum()
            else:
                per_class_acc[class_name] = 0.0
        
        return cm, per_class_acc


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: Optional[str] = None):
    """
    Plot confusion matrix with enhanced visualization.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
    """
    plt.figure(figsize=(16, 14))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Accuracy'}, linewidths=0.5)
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_per_class_accuracy(per_class_acc: Dict[str, float], save_path: Optional[str] = None):
    """
    Plot per-class accuracy as a bar chart.
    
    Args:
        per_class_acc: Dictionary of class names to accuracy values
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    classes = list(per_class_acc.keys())
    accuracies = [per_class_acc[c] * 100 for c in classes]
    
    # Create color gradient based on accuracy
    colors = plt.cm.RdYlGn([acc/100 for acc in accuracies])
    
    bars = ax.bar(classes, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add mean line
    mean_acc = np.mean(accuracies)
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean Accuracy: {mean_acc:.2f}%')
    
    ax.set_xlabel('Sign Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Per-class accuracy chart saved to {save_path}")
    
    plt.show()


def generate_all_graphs(
    trainer: Trainer,
    test_loader: DataLoader,
    class_names: List[str],
    save_dir: str = 'GRAPHS'
):
    """
    Generate all required graphs and save them.
    
    Args:
        trainer: Trained Trainer instance
        test_loader: Test data loader
        class_names: List of class names
        save_dir: Directory to save all graphs
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE TRAINING GRAPHS")
    print("="*60)
    
    # 1. Training Progress Graphs
    print("\n1. Generating Training Progress Graphs...")
    trainer.plot_learning_curves(save_path=os.path.join(save_dir, '01_training_progress.png'))
    
    # 2. Confusion Matrix
    print("\n2. Generating Confusion Matrix...")
    cm, per_class_acc = trainer.evaluate_with_confusion_matrix(test_loader, class_names)
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(save_dir, '02_confusion_matrix.png'))
    
    # 3. Per-Class Accuracy Bar Chart
    print("\n3. Generating Per-Class Accuracy Chart...")
    plot_per_class_accuracy(per_class_acc, save_path=os.path.join(save_dir, '03_per_class_accuracy.png'))
    
    # Calculate overall test accuracy
    overall_acc = np.trace(cm) / np.sum(cm) * 100
    print("\n" + "="*60)
    print(f"OVERALL TEST ACCURACY: {overall_acc:.4f}%")
    print("="*60)
    print(f"\nAll graphs saved to: {save_dir}/")
    
    return cm, per_class_acc
    
    # Print per-class summary
    print("\nPER-CLASS ACCURACY SUMMARY:")
    print("-" * 40)
    sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    for class_name, acc in sorted_acc:
        print(f"  {class_name:15s}: {acc*100:6.2f}%")
    
    return cm, per_class_acc


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
