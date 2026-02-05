#!/usr/bin/env python3
"""
Generate Confusion Matrix and Per-Class Accuracy graphs RIGHT NOW
"""
import torch
from torch.utils.data import DataLoader
from src.train import Trainer, plot_confusion_matrix, plot_per_class_accuracy
from src.dataset import ASLDataset
from src.model import ASLClassifier
from torchvision import transforms
import os
import numpy as np

# Configuration
SAVE_DIR = 'models'
GRAPH_DIR = 'GRAPHS'
BATCH_SIZE = 64

# ASL alphabet classes
ASL_CLASSES = [
    'A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

def main():
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation dataset
    print("\n📂 Loading validation dataset...")
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = ASLDataset(
        root_dir='data/asl_alphabet/asl_alphabet_train/asl_alphabet_train',
        transform=val_transform,
        classes=ASL_CLASSES
    )
    
    # Take only first 10000 samples for faster evaluation
    print(f"Total samples: {len(val_dataset)}")
    print("Using first 10000 samples for quick evaluation...")
    from torch.utils.data import Subset
    val_dataset = Subset(val_dataset, range(min(10000, len(val_dataset))))
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Using {len(val_dataset)} samples")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ASLClassifier(num_classes=len(ASL_CLASSES)).to(device)
    
    # Create trainer
    trainer = Trainer(model=model, device=device)
    
    # Load best model
    print("\n📥 Loading best model...")
    best_model_path = os.path.join(SAVE_DIR, 'asl_cnn_best.pt')
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found at: {best_model_path}")
        return
    
    trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✓ Loaded: {best_model_path}")
    
    # Create GRAPHS directory
    os.makedirs(GRAPH_DIR, exist_ok=True)
    
    # Generate Confusion Matrix and Per-Class Accuracy
    print("\n📊 Evaluating model and generating graphs...")
    print("This may take a few minutes...")
    
    cm, per_class_acc = trainer.evaluate_with_confusion_matrix(val_loader, ASL_CLASSES)
    
    print("\n2. Generating Confusion Matrix...")
    plot_confusion_matrix(cm, ASL_CLASSES, save_path=os.path.join(GRAPH_DIR, '02_confusion_matrix.png'))
    
    print("\n3. Generating Per-Class Accuracy Chart...")
    plot_per_class_accuracy(per_class_acc, save_path=os.path.join(GRAPH_DIR, '03_per_class_accuracy.png'))
    
    # Calculate overall accuracy
    overall_acc = np.trace(cm) / np.sum(cm) * 100
    print("\n" + "="*60)
    print(f"OVERALL ACCURACY: {overall_acc:.4f}%")
    print("="*60)
    print(f"\n✅ Graphs saved to: {GRAPH_DIR}/")
    print("\nNote: Training Progress graph will be generated when training completes.")

if __name__ == '__main__':
    main()
