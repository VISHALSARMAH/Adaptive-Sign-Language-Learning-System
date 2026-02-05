"""
Enhanced Training Script for ASL Model
Trains the model with early stopping at high accuracy and generates comprehensive graphs.
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import get_dataloaders, ASL_CLASSES
from src.model import ASLClassifier, export_to_onnx
from src.train import Trainer, generate_all_graphs

def main():
    # Configuration
    DATA_ROOT = 'data/asl_alphabet'
    BATCH_SIZE = 64
    VAL_SPLIT = 0.1
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    HIGH_ACCURACY_THRESHOLD = 0.9998  # 99.98%
    LEARNING_RATE = 1e-4
    SAVE_DIR = 'models'
    GRAPH_DIR = 'GRAPHS'
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("="*70)
    print("ASL MODEL TRAINING - ENHANCED VERSION")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"High accuracy threshold: {HIGH_ACCURACY_THRESHOLD*100:.2f}%")
    print("="*70)
    
    # Load data
    print("\n📦 Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        num_workers=0  # Use 0 to avoid multiprocessing issues on MPS
    )
    
    print(f"✓ Training samples: {len(train_loader.dataset)}")
    print(f"✓ Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"✓ Test samples: {len(test_loader.dataset)}")
    else:
        print("✓ Test samples: Using validation set for testing")
    print(f"✓ Number of classes: {len(ASL_CLASSES)}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ASLClassifier(
        num_classes=len(ASL_CLASSES),
        pretrained=True,
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\n🎓 Initializing trainer...")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-4,
        label_smoothing=0.1
    )
    
    # Train model
    print("\n🚀 Starting training...")
    print("-"*70)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        save_dir=SAVE_DIR,
        high_accuracy_threshold=HIGH_ACCURACY_THRESHOLD
    )
    
    # Load best model
    print("\n📥 Loading best model for evaluation...")
    best_model_path = os.path.join(SAVE_DIR, 'asl_cnn_best.pt')
    trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✓ Loaded: {best_model_path}")
    
    # Generate all graphs
    print("\n📊 Generating comprehensive graphs...")
    # Use validation set if test set is not available
    eval_loader = test_loader if test_loader else val_loader
    generate_all_graphs(
        trainer=trainer,
        test_loader=eval_loader,
        class_names=ASL_CLASSES,
        save_dir=GRAPH_DIR
    )
    
    # Export to ONNX
    print("\n📦 Exporting model to ONNX format...")
    onnx_path = os.path.join(SAVE_DIR, 'asl_cnn_best.onnx')
    export_to_onnx(trainer.model, onnx_path)
    print(f"✓ ONNX model saved: {onnx_path}")
    
    # Save training history
    import json
    import numpy as np
    
    print("\n💾 Saving training history...")
    history_data = {
        'train_loss': [float(x) for x in history['train_loss']],
        'train_acc': [float(x) for x in history['train_acc']],
        'val_loss': [float(x) for x in history['val_loss']],
        'val_acc': [float(x) for x in history['val_acc']],
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
        'overall_test_accuracy': float(np.trace(cm) / np.sum(cm)),
        'total_epochs': len(history['train_loss']),
        'best_val_accuracy': float(max(history['val_acc']))
    }
    
    history_path = os.path.join(SAVE_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"✓ Training history saved: {history_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Best Validation Accuracy: {max(history['val_acc'])*100:.4f}%")
    print(f"Test Accuracy: {(np.trace(cm) / np.sum(cm) * 100):.4f}%")
    print(f"Total Epochs: {len(history['train_loss'])}")
    print(f"Graphs saved to: {GRAPH_DIR}/")
    print(f"Models saved to: {SAVE_DIR}/")
    print("="*70)

if __name__ == "__main__":
    main()
