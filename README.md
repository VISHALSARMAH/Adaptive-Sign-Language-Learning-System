# 🤟 ASL-Tutor: Adaptive Sign Language Learning System

An end-to-end adaptive sign language learning system that uses deep learning for ASL recognition and contextual bandits for personalized curriculum adaptation.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

### Core Functionality
- **🧠 CNN-based ASL Recognition**: MobileNetV3-Small backbone trained on ASL Alphabet dataset
- **🎯 Adaptive Learning**: Contextual bandit (Thompson Sampling) selects optimal signs to practice
- **📈 Student Mastery Tracking**: Per-sign mastery with exponential moving averages
- **🌐 Web Interface**: Real-time webcam-based practice with instant feedback

### User Interface
- **📊 Session Reports**: Detailed performance reports after each practice session showing accuracy, time spent, and per-sign breakdown
- **🛑 Stop Button**: End practice anytime and view your session summary
- **⏭️ Skip Sign**: Skip to the next sign if you want to move on

### Accessibility
- **🔊 Audio Toggle**: Text-to-speech for instructions and feedback (clear ON/OFF visual indicator)
- **◐ High Contrast Mode**: Dark background with bright colors for better visibility
- **🐢 Slow Mode**: Extended countdown and feedback display times

### Research Tools
- **📉 A/B Testing Framework**: Compare adaptive vs random curriculum
- **📝 Session Logging**: Track all learning interactions
- **📊 Learning Analytics**: Generate learner reports and visualizations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Webcam (for practice sessions)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Step 1: Clone and Setup Environment

```bash
# Navigate to project directory
cd /Users/vishalsarmah/Desktop/Cap2

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download the Dataset (For Training Only)

Download the ASL Alphabet dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
- Extract to `data/asl_alphabet/` directory
- Expected structure:
  ```
  data/asl_alphabet/
  └── asl_alphabet_train/
      └── asl_alphabet_train/
          ├── A/
          ├── B/
          ├── C/
          ... (all letters)
          ├── del/
          ├── nothing/
          └── space/
  ```

### Step 4: Train the Model (Optional)

Skip this step if you already have `models/asl_cnn_best.pt`.

**Option A: Using Jupyter Notebook (Recommended)**
```bash
jupyter notebook notebooks/01_train_model.ipynb
```

**Option B: Using Command Line**
```bash
python -m src.train
```

### Step 5: Start the Backend Server

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Start FastAPI server
python -m src.api
```

✅ The API server will run at **http://localhost:8000**

### Step 6: Start the Frontend Server

Open a **new terminal window**:

```bash
cd /Users/vishalsarmah/Desktop/Cap2
python3 -m http.server 3000 --directory frontend
```

✅ The frontend will be available at **http://localhost:3000**

### Step 7: Start Learning!

1. Open **http://localhost:3000** in your browser
2. Allow camera permissions when prompted
3. Enter your username and click **"Start Learning"**
4. Practice signs following the on-screen prompts
5. Click **"Stop Practice"** to end session and view your performance report

---

## 📊 Model Performance

### Training Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | ~99.98% |
| **Number of Classes** | 29 |
| **Training Epochs** | 10 |
| **Best Model Checkpoint** | `models/asl_cnn_best.pt` |

### Architecture Details

| Component | Specification |
|-----------|---------------|
| **Backbone** | MobileNetV3-Small (pretrained on ImageNet) |
| **Input Size** | 224 × 224 RGB images |
| **Feature Extractor** | Frozen pretrained layers |
| **Classifier Head** | Global Avg Pool → FC(128) → ReLU → Dropout(0.2) → FC(29) |
| **Total Parameters** | ~1.5M (trainable: ~50K) |
| **Model File Size** | ~4 MB |

### Supported Signs

The model recognizes **29 classes**:
- **Letters**: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Special**: `space`, `del`, `nothing`

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Data Augmentation | RandomHorizontalFlip, ColorJitter, RandomRotation, RandomAffine |

### Inference Performance

| Metric | Value |
|--------|-------|
| Average Inference Time | ~50-100ms per image |
| Minimum Confidence Threshold | 30% (configurable) |
| Real-time Capable | ✅ Yes |

---

## 📁 Project Structure

```
ASL-Tutor/
├── src/
│   ├── __init__.py       # Package initialization
│   ├── api.py            # FastAPI backend server
│   ├── bandit.py         # Contextual bandit policies (Thompson Sampling)
│   ├── dataset.py        # Data loading and preprocessing
│   ├── evaluation.py     # A/B testing & research helpers
│   ├── inference.py      # Inference utilities and webcam demo
│   ├── model.py          # CNN model architecture
│   ├── student_model.py  # Student mastery tracking
│   └── train.py          # Training utilities
├── notebooks/
│   ├── 01_train_model.ipynb        # Model training notebook
│   └── 02_demo_and_evaluation.ipynb # Demo and evaluation notebook
├── frontend/
│   └── index.html        # Web-based tutor interface
├── models/
│   └── asl_cnn_best.pt   # Trained model weights
├── data/
│   ├── asl_alphabet/     # Dataset (download from Kaggle)
│   └── users/            # User progress data (JSON files)
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── FUTURE_UPDATES.md     # Planned improvements
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check - returns server status |
| `/signs` | GET | List all available ASL signs |
| `/predict` | POST | Predict sign from base64 image |
| `/next_sign` | POST | Get next sign to practice (bandit selection) |
| `/update` | POST | Update progress after attempt |
| `/progress/{user_id}` | GET | Get student's learning progress |
| `/session/start` | POST | Start a new learning session |
| `/leaderboard` | GET | Get top learners by mastery |

### Example API Usage

```python
import requests
import base64

# Predict a sign
with open('hand_image.jpg', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:8000/predict', json={
    'image_base64': image_base64,
    'user_id': 'student1',
    'target_sign': 'A'
})

print(response.json())
# {'predicted_sign': 'A', 'confidence': 0.98, 'is_correct': True, ...}
```

---

## 🎯 Contextual Bandit Algorithm

The adaptive curriculum uses **Linear Thompson Sampling** to personalize sign selection:

### Context Features (per sign)
1. Current mastery level (0-1)
2. Normalized attempt count
3. Average response time
4. Days since last practice
5. Overall learner mastery
6. Current streak

### Reward Signal
| Outcome | Reward |
|---------|--------|
| Correct & Fast (< 3s) | 1.0 |
| Correct & Slow | 0.5-1.0 (scaled) |
| Incorrect | 0.0 |

### Selection Strategy
- Samples from posterior distribution
- Adds mastery-based bonus to prioritize weak signs
- Balances exploration (new signs) and exploitation (practice weak signs)

---

## ♿ Accessibility Features

| Feature | Description | Toggle |
|---------|-------------|--------|
| **🔊 Audio** | Text-to-speech for instructions and feedback | Click "Audio ON/OFF" button |
| **◐ Contrast** | High contrast mode with dark background | Click "Contrast" button |
| **🐢 Slow Mode** | Extended timers for users who need more time | Click "Slow" button |

---

## 🔬 Research & Evaluation

### A/B Testing

Compare adaptive vs random curriculum:

```python
from src.evaluation import ABTestManager

ab = ABTestManager()
results = ab.run_ab_experiment(n_users_per_group=10, n_steps_per_user=200)
ab.analyze_results(results)
ab.plot_results(results)
```

### Session Logging

```python
from src.evaluation import SessionLogger

logger = SessionLogger()
logger.start_session(user_id='student1', mode='adaptive')
# ... log attempts ...
logger.end_session()
```

### Learner Reports

```python
from src.evaluation import print_learner_report
print_learner_report('student1')
```

---

## 🛠️ Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Camera not working** | Allow camera permissions in browser settings |
| **Model not loading** | Ensure `models/asl_cnn_best.pt` exists |
| **API not responding** | Check if backend server is running on port 8000 |
| **CORS errors** | Make sure frontend is served via HTTP server, not file:// |
| **Slow predictions** | Close other resource-intensive applications |

### Checking Server Status

```bash
# Check if API is running
curl http://localhost:8000/

# Expected response:
# {"message":"ASL-Tutor API is running!","version":"1.0.0"}
```

---

## 📋 Requirements

### System Requirements
- **OS**: macOS, Linux, or Windows
- **Python**: 3.10+
- **RAM**: 4GB minimum, 8GB recommended
- **Webcam**: Required for practice sessions

### Python Dependencies
- PyTorch 2.0+
- FastAPI
- Uvicorn
- Pillow
- NumPy
- OpenCV (for standalone webcam demo)

See `requirements.txt` for full list.

---

## 🔮 Future Updates

See [FUTURE_UPDATES.md](FUTURE_UPDATES.md) for planned improvements including:
- Early stopping with model rollback during training
- Training visualization graphs
- Learning rate scheduling
- Additional data augmentation techniques

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{asl_tutor_2026,
  title={ASL-Tutor: Adaptive Sign Language Learning System},
  author={Vishal Sarmah},
  year={2026},
  description={CNN-based ASL recognition with contextual bandit curriculum adaptation},
  url={https://github.com/vishalsarmah/asl-tutor}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **ASL Alphabet Dataset**: [Kaggle - ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **MobileNetV3**: Howard et al., "Searching for MobileNetV3" (2019)
- **Thompson Sampling**: Thompson, "On the likelihood that one unknown probability exceeds another" (1933)
