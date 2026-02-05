# 🤟 ASL-Tutor: Adaptive Sign Language Learning System

A state-of-the-art sign language learning platform combining deep learning with adaptive curriculum personalization. The system achieves **99.99% validation accuracy** using a highly optimized MobileNetV3-based CNN and employs contextual bandits for intelligent sign selection.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-99.99%25-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Key Features

### 🎯 High-Performance ASL Recognition
- **99.99% Validation Accuracy** on 29 ASL alphabet signs
- Real-time inference (~50-100ms per prediction)
- MobileNetV3-Small architecture optimized for speed and accuracy
- Enhanced training with early stopping and intelligent model checkpointing

### 🧠 Adaptive Learning System
- **Contextual Bandit Algorithm**: Thompson Sampling for personalized sign selection
- **Mastery Tracking**: Real-time progress monitoring per sign
- **Intelligent Curriculum**: Adapts to each learner's strengths and weaknesses
- **Session Analytics**: Detailed performance reports with visualizations

### 🌐 Interactive Web Interface
- Real-time webcam-based practice
- Instant feedback with confidence scores
- Session summaries with detailed breakdowns
- Stop/skip functionality for flexible learning

### ♿ Accessibility Features
- **Audio Toggle**: Text-to-speech instructions and feedback
- **High Contrast Mode**: Enhanced visibility for users with visual impairments
- **Slow Mode**: Extended timers for learners needing extra time

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ 
- Webcam (for practice sessions)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation & Setup

```bash
# 1. Navigate to project directory
cd /Users/vishalsarmah/Desktop/Cap2

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the Application

**Start Backend Server:**
```bash
./run_server.sh
# or manually: python -m uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Start Frontend Server (in new terminal):**
```bash
./run_frontend.sh
# or manually: cd frontend && python3 -m http.server 3000
```

**Access the Application:**
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs

### Training the Model (Optional)

The project includes a pre-trained model with 99.99% accuracy. To retrain:

```bash
# Using the enhanced training script
python train_enhanced.py

# Or use the Jupyter notebook
jupyter notebook notebooks/01_train_model.ipynb
```

**Training Features:**
- Early stopping at 99.98% accuracy threshold
- Automatic model checkpointing per epoch
- Best model selection and saving
- Comprehensive training visualizations (graphs saved to `GRAPHS/`)

---

## 🤖 Model Architecture & Performance

### CNN Architecture

**Base Model:** MobileNetV3-Small (Pretrained on ImageNet)
- Lightweight and efficient architecture designed for mobile and edge devices
- Optimized with inverted residual blocks and squeeze-and-excitation layers
- Total parameters: ~1,004,605 (trainable: ~50,000)

**Custom Classifier:**
```
Input (224×224 RGB) 
    → MobileNetV3 Feature Extractor (frozen pretrained weights)
    → Global Average Pooling
    → Fully Connected (128 neurons)
    → ReLU Activation
    → Dropout (0.2)
    → Fully Connected (29 classes)
    → Softmax
```

### Training Performance

| Metric | Value |
|--------|-------|
| **Final Validation Accuracy** | **99.9885%** |
| **Training Epochs** | 4 (early stopping triggered) |
| **Best Epoch** | Epoch 3 |
| **Dataset Size** | 87,000 images (29 classes) |
| **Batch Size** | 64 |
| **Learning Rate** | 0.0001 (AdamW optimizer) |
| **Early Stopping Threshold** | 99.98% |

**Training Features:**
- Label Smoothing Cross-Entropy (smoothing=0.1)
- ReduceLROnPlateau scheduler (factor=0.1, patience=3)
- Enhanced early stopping with accuracy drop detection
- Automatic per-epoch model checkpointing
- Best model selection and preservation

### Supported Signs (29 Classes)

**Letters:** A-Z (26 letters)  
**Special Signs:** `space`, `del` (delete), `nothing`

### Inference Performance

| Metric | Value |
|--------|-------|
| **Inference Time** | 50-100ms per image |
| **Confidence Threshold** | 30% (configurable) |
| **Real-time Processing** | ✅ Yes (MPS/CUDA accelerated) |
| **Model Size** | 4.0 MB |

### Training Visualizations

Comprehensive graphs generated during training (saved in `GRAPHS/`):

1. **Training Progress** (`01_training_progress.png`)
   - Loss curves (training & validation)
   - Accuracy curves (training & validation)

2. **Confusion Matrix** (`02_confusion_matrix.png`)
   - 29×29 normalized confusion matrix
   - Per-class prediction analysis

3. **Per-Class Accuracy** (`03_per_class_accuracy.png`)
   - Color-coded bar chart showing accuracy for each sign
   - Identifies strong and weak classes

---

## 📁 Project Structure

```
ASL-Tutor/
├── src/                      # Core application code
│   ├── api.py               # FastAPI backend server
│   ├── model.py             # MobileNetV3 CNN architecture
│   ├── train.py             # Enhanced training with early stopping
│   ├── dataset.py           # Data loading and augmentation
│   ├── inference.py         # Real-time inference utilities
│   ├── bandit.py            # Thompson Sampling curriculum
│   ├── student_model.py     # Learner progress tracking
│   └── evaluation.py        # A/B testing & analytics
│
├── frontend/                 # Web interface
│   └── index.html           # Single-page application
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_train_model.ipynb
│   └── 02_demo_and_evaluation.ipynb
│
├── models/                  # Trained models
│   ├── asl_cnn_best.pt     # Best model (99.99% accuracy)
│   └── asl_cnn_epoch_*.pt  # Epoch checkpoints
│
├── GRAPHS/                  # Training visualizations
│   ├── 01_training_progress.png
│   ├── 02_confusion_matrix.png
│   └── 03_per_class_accuracy.png
│
├── data/                    # Datasets and user data
│   ├── asl_alphabet/       # ASL training images
│   └── users/              # User progress (JSON)
│
├── train_enhanced.py        # Enhanced training script
├── run_server.sh           # Backend startup script
├── run_frontend.sh         # Frontend startup script
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and server status |
| `/predict` | POST | Predict ASL sign from base64 image |
| `/next_sign` | POST | Get next sign (adaptive selection) |
| `/update` | POST | Update learner progress |
| `/progress/{user_id}` | GET | Retrieve user progress data |

**Example Usage:**
```python
import requests
import base64

# Predict a sign
response = requests.post('http://localhost:8000/predict', json={
    'image_base64': image_base64,
    'user_id': 'student1',
    'target_sign': 'A'
})
# Returns: {'predicted_sign': 'A', 'confidence': 0.98, 'is_correct': True}
```

---

## 🎯 Adaptive Learning Algorithm

The system employs **Linear Thompson Sampling**, a contextual bandit algorithm, to personalize sign selection for each learner.

### How It Works

1. **Context Features** (per sign):
   - Current mastery level (0-1 scale)
   - Attempt count and success rate
   - Average response time
   - Days since last practice
   - Learning velocity and streak

2. **Reward Signal**:
   - Correct & Fast (< 3s): 1.0
   - Correct & Slow: 0.5-1.0 (time-scaled)
   - Incorrect: 0.0

3. **Selection Strategy**:
   - Samples from Bayesian posterior distribution
   - Balances exploration (new signs) vs exploitation (practice weak areas)
   - Prioritizes signs with lower mastery for faster improvement

This approach ensures learners practice signs they need most while maintaining engagement through variety.

---

## 🛠️ Technical Details

### System Requirements
- **OS**: macOS, Linux, or Windows 10+
- **Python**: 3.10 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for models and dependencies
- **Webcam**: Required for practice sessions
- **GPU**: Optional (MPS/CUDA accelerated when available)

### Key Dependencies
- PyTorch 2.0+ (Deep Learning)
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Pillow (Image processing)
- NumPy (Numerical computing)
- Matplotlib & Seaborn (Visualizations)

### Performance Optimization
- **Model**: MobileNetV3 architecture for fast inference
- **Hardware Acceleration**: Automatic MPS/CUDA detection
- **Inference**: Optimized for real-time processing
- **Memory**: Efficient batch processing and caching

---

## � Recent Updates

### Latest Enhancements (February 2026)

✅ **Enhanced Training Pipeline**
- Implemented intelligent early stopping (99.98% threshold)
- Automatic per-epoch model checkpointing
- Best model selection and preservation
- Training converged in 4 epochs with 99.9885% accuracy

✅ **Comprehensive Visualizations**
- Training progress graphs (loss & accuracy curves)
- 29×29 normalized confusion matrix
- Per-class accuracy analysis
- All graphs saved at 300 DPI for publication quality

✅ **Optimized Model Architecture**
- Fine-tuned MobileNetV3-Small backbone
- Label smoothing for better generalization
- ReduceLROnPlateau scheduler for adaptive learning rates
- AdamW optimizer with weight decay

---

## 🔮 Future Enhancements

- Multi-word sign language phrase recognition
- Real-time sign language translation
- Mobile application (iOS/Android)
- Support for additional sign languages (BSL, JSL, etc.)
- Advanced analytics dashboard
- Gamification features (badges, achievements)

---

## 🙏 Acknowledgments

- **Dataset**: [ASL Alphabet Dataset on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Architecture**: MobileNetV3 by Howard et al. (2019)
- **Algorithm**: Thompson Sampling by Thompson (1933)

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Vishal Sarmah**  
Building adaptive learning systems with AI

---

**⭐ If you find this project useful, please consider giving it a star!**
