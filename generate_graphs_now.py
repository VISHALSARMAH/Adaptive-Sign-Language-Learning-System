#!/usr/bin/env python3
"""
Generate all graphs from the best model immediately
"""
import torch
from torch.utils.data import DataLoader
from src.train import Trainer, generate_all_graphs
from src.dataset import ASLDataset
from src.model import ASLClassifier
import os

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
    from torchvision import transforms
    
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
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = ASLClassifier(num_classes=len(ASL_CLASSES)).to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device
    )
    
    # Load best model
    print("\n📥 Loading best model...")
    best_model_path = os.path.join(SAVE_DIR, 'asl_cnn_best.pt')
    if not os.path.exists(best_model_path):
        print(f"❌ Best model not found at: {best_model_path}")
        return
    
    trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"✓ Loaded: {best_model_path}")
    
    # Generate all graphs
    print("\n📊 Generating all graphs...")
    generate_all_graphs(
        trainer=trainer,
        test_loader=val_loader,
        class_names=ASL_CLASSES,
        save_dir=GRAPH_DIR
    )
    
    print("\n✅ All graphs generated successfully!")

if __name__ == '__main__':
    main()
