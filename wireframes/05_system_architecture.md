# ASL-Tutor System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ASL-TUTOR SYSTEM ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND LAYER                                │
└─────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────┐
    │                   Web Browser (Client)                     │
    │  ──────────────────────────────────────────────────────    │
    │                                                             │
    │  ┌──────────────────────────────────────────────────────┐ │
    │  │  HTML5 / CSS3 / JavaScript                           │ │
    │  │  ────────────────────────────────────────────────    │ │
    │  │                                                       │ │
    │  │  • Single Page Application (index.html)              │ │
    │  │  • WebRTC for webcam access                          │ │
    │  │  • Canvas API for image capture                      │ │
    │  │  • Fetch API for REST calls                          │ │
    │  │  • LocalStorage for client-side caching              │ │
    │  │  • Web Speech API for text-to-speech                 │ │
    │  │                                                       │ │
    │  └──────────────────────────────────────────────────────┘ │
    │                                                             │
    │  📹 Webcam → Canvas → Base64 Image → HTTP POST            │
    │                                                             │
    └──────────────────────┬──────────────────────────────────────┘
                           │
                           │ HTTP/HTTPS
                           │ (JSON REST API)
                           │
                           ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                          BACKEND LAYER                                  │
└─────────────────────────────────────────────────────────────────────────┘

    ┌────────────────────────────────────────────────────────────┐
    │              FastAPI Application Server                    │
    │              (src/api.py)                                  │
    │  ──────────────────────────────────────────────────────    │
    │                                                             │
    │  Endpoints:                                                 │
    │  ┌──────────────────────────────────────────────────────┐ │
    │  │  GET  /                  → Health check              │ │
    │  │  POST /predict           → Sign recognition          │ │
    │  │  POST /next_sign         → Bandit sign selection     │ │
    │  │  POST /update            → Update student progress   │ │
    │  │  GET  /progress/{user}   → Retrieve progress         │ │
    │  └──────────────────────────────────────────────────────┘ │
    │                                                             │
    │  Middleware:                                                │
    │  • CORS handling                                            │
    │  • Request validation (Pydantic)                            │
    │  • Error handling & logging                                 │
    │                                                             │
    └──────┬──────────────────┬─────────────────┬────────────────┘
           │                  │                 │
           │                  │                 │
           ▼                  ▼                 ▼

┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  ML Inference    │  │  Bandit Engine  │  │  Student Model   │
│  (src/inference) │  │  (src/bandit)   │  │  (src/student)   │
│  ──────────────  │  │  ──────────────  │  │  ──────────────  │
│                  │  │                  │  │                  │
│  • Image decode  │  │  • Thompson      │  │  • Mastery EMA   │
│  • Preprocessing │  │    Sampling      │  │  • Attempt logs  │
│  • Model forward │  │  • Context       │  │  • Time tracking │
│  • Softmax       │  │    features      │  │  • Streak calc   │
│  • Argmax        │  │  • Reward calc   │  │  • JSON persist  │
│                  │  │  • Exploration   │  │                  │
└────────┬─────────┘  └──────────────────┘  └────────┬─────────┘
         │                                            │
         │                                            │
         ▼                                            ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                         MODEL & DATA LAYER                              │
└─────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────┐        ┌──────────────────────┐
    │   Deep Learning Model         │        │   User Data Storage  │
    │   (models/asl_cnn_best.pt)    │        │   (data/users/)      │
    │   ──────────────────────────   │        │   ──────────────────  │
    │                                │        │                      │
    │  Architecture:                 │        │  Format: JSON        │
    │  ┌──────────────────────────┐ │        │                      │
    │  │  Input: 224×224×3 RGB   │ │        │  Files:              │
    │  │         ↓                │ │        │  • student1.json     │
    │  │  MobileNetV3-Small       │ │        │  • student2.json     │
    │  │  (pretrained backbone)   │ │        │  • ...               │
    │  │         ↓                │ │        │                      │
    │  │  Global Avg Pool         │ │        │  Content:            │
    │  │         ↓                │ │        │  • Sign mastery      │
    │  │  FC(128) + ReLU          │ │        │  • Attempt history   │
    │  │         ↓                │ │        │  • Session logs      │
    │  │  Dropout(0.2)            │ │        │  • Timestamps        │
    │  │         ↓                │ │        │  • Bandit params     │
    │  │  FC(29) + Softmax        │ │        │                      │
    │  │         ↓                │ │        └──────────────────────┘
    │  │  Output: 29 classes      │ │
    │  └──────────────────────────┘ │
    │                                │
    │  Performance:                  │
    │  • Accuracy: 99.9885%          │
    │  • Inference: 50-100ms         │
    │  • Size: 4.0 MB                │
    │  • Device: MPS/CUDA/CPU        │
    │                                │
    └────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      TRAINING & VISUALIZATION LAYER                     │
└─────────────────────────────────────────────────────────────────────────┘

    ┌───────────────────────────────┐        ┌──────────────────────┐
    │   Training Pipeline           │        │   Visualizations     │
    │   (train_enhanced.py)         │        │   (GRAPHS/)          │
    │   ──────────────────────────   │        │   ──────────────────  │
    │                                │        │                      │
    │  • Dataset loading             │        │  Generated Graphs:   │
    │  • Data augmentation           │        │                      │
    │  • Model initialization        │        │  1. Training         │
    │  • Training loop               │        │     Progress         │
    │  • Early stopping (99.98%)     │        │     - Loss curves    │
    │  • Per-epoch checkpoints       │        │     - Acc curves     │
    │  • Best model selection        │        │                      │
    │  • Graph generation            │   ─────┤  2. Confusion        │
    │                                │        │     Matrix (29×29)   │
    │  Training Config:              │        │                      │
    │  • AdamW optimizer             │        │  3. Per-class        │
    │  • Label smoothing (0.1)       │        │     Accuracy         │
    │  • LR scheduler                │        │     Bar chart        │
    │  • Batch size: 64              │        │                      │
    │                                │        └──────────────────────┘
    └────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW DIAGRAM                              │
└─────────────────────────────────────────────────────────────────────────┘

    User Webcam
         │
         │ (1) Capture frame
         ▼
    Base64 Encode
         │
         │ (2) POST /predict
         ▼
    FastAPI Server
         │
         ├──(3a)─→ Decode Image ──→ Preprocess ──→ CNN Model
         │                                              │
         │                                              │ (4) Prediction
         │                                              ▼
         │         ┌─────────────────────── Softmax Probabilities
         │         │                                    │
         │         │ (5) argmax + confidence            │
         │         ▼                                    │
         ├──(3b)─→ Student Model ←──────────────────────┘
         │              │
         │              │ (6) Update mastery
         │              ▼
         │         Save to JSON
         │              │
         ├──(3c)─→ Bandit Algorithm
         │              │
         │              │ (7) Calculate reward
         │              │ (8) Update posterior
         │              │ (9) Select next sign
         │              ▼
         │         Return next_sign
         │              │
         └──────────────┘
                        │
         (10) JSON Response
                        │
                        ▼
                  Web Browser
                        │
                        │ (11) Display feedback
                        │ (12) Show next sign
                        ▼
                    User sees result


┌─────────────────────────────────────────────────────────────────────────┐
│                      TECHNOLOGY STACK SUMMARY                           │
└─────────────────────────────────────────────────────────────────────────┘

    Frontend:           Backend:              ML/AI:
    ─────────          ─────────              ──────
    • HTML5            • Python 3.10+         • PyTorch 2.0+
    • CSS3             • FastAPI              • MobileNetV3
    • JavaScript       • Uvicorn              • Transfer Learning
    • WebRTC           • Pydantic              • AdamW Optimizer
    • Canvas API       • CORS middleware       • Label Smoothing
    
    Data:              DevOps:                Training:
    ─────              ──────                 ─────────
    • JSON files       • Shell scripts        • Early stopping
    • LocalStorage     • Virtual env          • Checkpointing
    • User profiles    • Git                  • Data augmentation
                       • Requirements.txt     • Matplotlib/Seaborn
```

## Key Components

### 1. **Frontend (Client)**
- Handles user interactions
- Captures webcam frames
- Sends images to backend
- Displays predictions and feedback

### 2. **Backend (Server)**
- FastAPI REST API
- Routes requests to appropriate modules
- Orchestrates inference, bandit, and student tracking

### 3. **ML Inference Engine**
- Loads trained PyTorch model
- Preprocesses images
- Returns predictions with confidence

### 4. **Contextual Bandit**
- Thompson Sampling algorithm
- Selects next optimal sign to practice
- Balances exploration/exploitation

### 5. **Student Model**
- Tracks mastery per sign
- Maintains learning history
- Persists to JSON files

### 6. **Training Pipeline**
- Trains CNN on ASL dataset
- Implements early stopping
- Generates performance graphs
- Saves model checkpoints
