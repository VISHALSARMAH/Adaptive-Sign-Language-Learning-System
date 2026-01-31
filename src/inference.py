"""
ASL Inference Module
====================
Inference utilities for the trained ASL classifier.
Includes webcam demo functionality.
"""

import os
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from .model import ASLClassifier, load_model
from .dataset import ASL_CLASSES, IMAGENET_MEAN, IMAGENET_STD, get_val_transforms


class ASLPredictor:
    """
    ASL Sign Predictor class for inference.
    
    Args:
        model_path: Path to trained model weights (.pt)
        model_type: 'mobilenet' or 'efficientnet'
        device: Device to run inference on
        use_onnx: Use ONNX runtime for inference (faster)
    """
    
    def __init__(
        self,
        model_path: str = 'models/asl_cnn.pt',
        model_type: str = 'mobilenet',
        device: str = None,
        use_onnx: bool = False
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_onnx = use_onnx
        self.classes = ASL_CLASSES
        
        if use_onnx:
            self._load_onnx_model(model_path.replace('.pt', '.onnx'))
        else:
            self.model = load_model(model_path, model_type, self.device)
        
        # Preprocessing transform
        self.transform = get_val_transforms(image_size=224)
    
    def _load_onnx_model(self, onnx_path: str):
        """Load ONNX model for inference."""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model input.
        
        Args:
            image: RGB image as numpy array (any size)
        
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
        """
        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    @torch.no_grad()
    def predict_sign(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict ASL sign from image.
        
        Args:
            image: RGB image as numpy array (224x224 preferred, any size accepted)
        
        Returns:
            (predicted_label, confidence) tuple
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        if self.use_onnx:
            # ONNX inference
            input_np = input_tensor.numpy()
            outputs = self.ort_session.run(None, {self.input_name: input_np})
            logits = torch.tensor(outputs[0])
        else:
            # PyTorch inference
            input_tensor = input_tensor.to(self.device)
            logits = self.model(input_tensor)
        
        # Get prediction
        probs = F.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)
        
        predicted_label = self.classes[predicted_idx.item()]
        confidence_score = confidence.item()
        
        return predicted_label, confidence_score
    
    def predict_batch(self, images: list) -> list:
        """
        Predict signs for a batch of images.
        
        Args:
            images: List of RGB images as numpy arrays
        
        Returns:
            List of (predicted_label, confidence) tuples
        """
        results = []
        for image in images:
            results.append(self.predict_sign(image))
        return results
    
    def get_top_k(self, image: np.ndarray, k: int = 5) -> list:
        """
        Get top-k predictions.
        
        Args:
            image: RGB image
            k: Number of top predictions
        
        Returns:
            List of (label, confidence) tuples sorted by confidence
        """
        input_tensor = self.preprocess(image)
        
        if self.use_onnx:
            input_np = input_tensor.numpy()
            outputs = self.ort_session.run(None, {self.input_name: input_np})
            logits = torch.tensor(outputs[0])
        else:
            input_tensor = input_tensor.to(self.device)
            logits = self.model(input_tensor)
        
        probs = F.softmax(logits, dim=1).squeeze()
        top_probs, top_indices = torch.topk(probs, k)
        
        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            results.append((self.classes[idx], prob))
        
        return results


def predict_sign(
    image: np.ndarray,
    predictor: Optional[ASLPredictor] = None,
    model_path: str = 'models/asl_cnn.pt'
) -> Tuple[str, float]:
    """
    Simple function to predict ASL sign from image.
    
    Args:
        image: 224x224 RGB image as numpy array
        predictor: Optional ASLPredictor instance (creates new if None)
        model_path: Path to model weights
    
    Returns:
        (predicted_label, confidence) tuple
    """
    if predictor is None:
        predictor = ASLPredictor(model_path=model_path)
    
    return predictor.predict_sign(image)


class WebcamDemo:
    """
    Real-time ASL recognition from webcam.
    
    Usage:
        demo = WebcamDemo(model_path='models/asl_cnn.pt')
        demo.run()
    """
    
    def __init__(
        self,
        model_path: str = 'models/asl_cnn.pt',
        model_type: str = 'mobilenet',
        camera_id: int = 0,
        crop_size: int = 300
    ):
        self.predictor = ASLPredictor(model_path=model_path, model_type=model_type)
        self.camera_id = camera_id
        self.crop_size = crop_size
    
    def run(self):
        """Run the webcam demo."""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("ASL Webcam Demo")
        print("===============")
        print("Position your hand in the green box")
        print("Press 'q' to quit")
        print()
        
        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # ROI (region of interest) for hand
        roi_x = (frame_width - self.crop_size) // 2
        roi_y = (frame_height - self.crop_size) // 2
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw ROI rectangle
            cv2.rectangle(
                frame,
                (roi_x, roi_y),
                (roi_x + self.crop_size, roi_y + self.crop_size),
                (0, 255, 0),
                2
            )
            
            # Extract ROI
            roi = frame[roi_y:roi_y + self.crop_size, roi_x:roi_x + self.crop_size]
            
            # Convert BGR to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            
            # Predict
            label, confidence = self.predictor.predict_sign(roi_rgb)
            
            # Draw prediction
            text = f"{label}: {confidence*100:.1f}%"
            cv2.putText(
                frame,
                text,
                (roi_x, roi_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Draw confidence bar
            bar_width = int(self.crop_size * confidence)
            cv2.rectangle(
                frame,
                (roi_x, roi_y + self.crop_size + 5),
                (roi_x + bar_width, roi_y + self.crop_size + 25),
                (0, 255, 0),
                -1
            )
            
            # Show frame
            cv2.imshow('ASL Recognition', frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


def run_webcam_demo(model_path: str = 'models/asl_cnn.pt'):
    """Run webcam demo (convenience function)."""
    demo = WebcamDemo(model_path=model_path)
    demo.run()


if __name__ == "__main__":
    # Test inference
    print("Testing ASL Inference...")
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # This would fail without a trained model, just testing the pipeline
    try:
        predictor = ASLPredictor(model_path='models/asl_cnn.pt')
        label, conf = predictor.predict_sign(dummy_image)
        print(f"Predicted: {label} ({conf*100:.1f}%)")
    except FileNotFoundError:
        print("No trained model found. Train the model first.")
