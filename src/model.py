"""
ASL CNN Model Module
====================
MobileNetV3-Small based classifier for ASL alphabet recognition.
Supports training, export to ONNX, and inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional

from .dataset import ASL_CLASSES


class ASLClassifier(nn.Module):
    """
    ASL Alphabet Classifier using MobileNetV3-Small backbone.
    
    Architecture:
    - MobileNetV3-Small pretrained on ImageNet
    - Global Average Pooling (from backbone)
    - Dense(128, ReLU, Dropout 0.2)
    - Dense(29, Softmax)
    
    Args:
        num_classes: Number of output classes (default: 29 for ASL)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate for classifier head
    """
    
    def __init__(
        self, 
        num_classes: int = 29,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super(ASLClassifier, self).__init__()
        
        self.num_classes = num_classes
        
        # Load MobileNetV3-Small backbone
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        backbone = models.mobilenet_v3_small(weights=weights)
        
        # Get the feature extractor (everything except classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Get the input features to the classifier
        # MobileNetV3-Small has 576 features after avgpool
        in_features = 576
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier weights using Xavier initialization."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and confidence scores.
        
        Returns:
            (predicted_labels, confidence_scores)
        """
        probs = self.predict_proba(x)
        confidence, labels = torch.max(probs, dim=1)
        return labels, confidence


class ASLClassifierEfficientNet(nn.Module):
    """
    Alternative: ASL Classifier using EfficientNet-B0 backbone.
    
    Use this if you want a slightly larger but more accurate model.
    """
    
    def __init__(
        self, 
        num_classes: int = 29,
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super(ASLClassifierEfficientNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Load EfficientNet-B0 backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        backbone = models.efficientnet_b0(weights=weights)
        
        # Get the feature extractor
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # EfficientNet-B0 has 1280 features
        in_features = 1280
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_classifier()
    
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = self.predict_proba(x)
        confidence, labels = torch.max(probs, dim=1)
        return labels, confidence


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    
    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # Convert to log probabilities
        log_preds = F.log_softmax(pred, dim=-1)
        
        # Compute smoothed loss
        loss = -log_preds.sum(dim=-1).mean()  # Uniform part
        nll = F.nll_loss(log_preds, target)   # Target part
        
        # Combine
        return (1 - self.smoothing) * nll + self.smoothing * loss / n_classes


def export_to_onnx(
    model: nn.Module,
    output_path: str = 'models/asl_cnn.onnx',
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    device: str = 'cpu'
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained PyTorch model
        output_path: Path to save ONNX model
        input_size: Input tensor shape (B, C, H, W)
        device: Device to use for export
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify the model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified successfully!")


def load_model(
    weights_path: str,
    model_type: str = 'mobilenet',
    device: str = 'cpu'
) -> nn.Module:
    """
    Load a trained ASL classifier.
    
    Args:
        weights_path: Path to .pt weights file
        model_type: 'mobilenet' or 'efficientnet'
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    if model_type == 'mobilenet':
        model = ASLClassifier(num_classes=len(ASL_CLASSES), pretrained=False)
    elif model_type == 'efficientnet':
        model = ASLClassifierEfficientNet(num_classes=len(ASL_CLASSES), pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing ASL Classifier...")
    
    # Create model
    model = ASLClassifier(num_classes=29, pretrained=True)
    print(f"Model created: ASLClassifier (MobileNetV3-Small)")
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test prediction
    labels, confidence = model.predict(dummy_input)
    print(f"Predicted labels: {labels}")
    print(f"Confidence: {confidence}")
    
    # Test alternative model
    print("\n--- EfficientNet-B0 Alternative ---")
    model_eff = ASLClassifierEfficientNet(num_classes=29, pretrained=True)
    print(f"Trainable parameters: {count_parameters(model_eff):,}")
