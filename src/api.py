"""
ASL-Tutor FastAPI Backend
=========================
REST API for the adaptive sign language learning system.

Endpoints:
- POST /predict: Predict sign from image
- POST /next_sign: Get next sign to practice
- POST /update: Update student model after attempt
- GET /progress/{user_id}: Get student progress
"""

import os
import base64
import time
from typing import Optional
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import ASLPredictor
from src.student_model import StudentModel
from src.bandit import LinearThompsonSampling, compute_reward, get_bandit_policy
from src.dataset import ASL_CLASSES


# =============================================================================
# FastAPI App Setup
# =============================================================================

app = FastAPI(
    title="ASL-Tutor API",
    description="Adaptive Sign Language Learning System",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State (in production, use proper state management)
# =============================================================================

# Model predictor (lazy loaded)
_predictor: Optional[ASLPredictor] = None

# Bandit policies per user (in production, use database)
_bandits: dict = {}

# Configuration
MODEL_PATH = os.environ.get('ASL_MODEL_PATH', 'models/asl_cnn_best.pt')
DATA_DIR = os.environ.get('ASL_DATA_DIR', 'data/users')


def get_predictor() -> ASLPredictor:
    """Get or create the model predictor."""
    global _predictor
    if _predictor is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail=f"Model not found at {MODEL_PATH}. Train the model first."
            )
        _predictor = ASLPredictor(model_path=MODEL_PATH)
    return _predictor


def get_bandit(user_id: str) -> LinearThompsonSampling:
    """Get or create bandit policy for user."""
    global _bandits
    if user_id not in _bandits:
        policy = LinearThompsonSampling()
        
        # Try to load existing policy
        policy_path = os.path.join(DATA_DIR, f'{user_id}_bandit.json')
        if os.path.exists(policy_path):
            policy.load(policy_path)
        
        _bandits[user_id] = policy
    
    return _bandits[user_id]


def save_bandit(user_id: str):
    """Save bandit policy for user."""
    if user_id in _bandits:
        policy_path = os.path.join(DATA_DIR, f'{user_id}_bandit.json')
        os.makedirs(os.path.dirname(policy_path), exist_ok=True)
        _bandits[user_id].save(policy_path)


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request for sign prediction."""
    image_base64: str  # Base64 encoded image
    user_id: str
    target_sign: Optional[str] = None  # Expected sign (for verification)


class PredictResponse(BaseModel):
    """Response for sign prediction."""
    predicted_sign: str
    confidence: float
    is_correct: Optional[bool] = None
    top_5: list = []


class NextSignRequest(BaseModel):
    """Request for next sign selection."""
    user_id: str


class NextSignResponse(BaseModel):
    """Response with next sign to practice."""
    sign: str
    sign_image_url: Optional[str] = None
    mastery: float
    attempts: int


class UpdateRequest(BaseModel):
    """Request to update student model."""
    user_id: str
    sign_label: str
    correct: bool
    response_time: float  # seconds


class UpdateResponse(BaseModel):
    """Response after update."""
    success: bool
    new_mastery: float
    overall_mastery: float


class ProgressResponse(BaseModel):
    """Student progress response."""
    user_id: str
    overall_mastery: float
    total_attempts: int
    total_sessions: int
    signs_mastered: int
    weak_signs: list
    progress_by_sign: dict


class SessionRequest(BaseModel):
    """Request to start/end session."""
    user_id: str
    mode: str = "adaptive"  # "adaptive" or "fixed"


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "ASL-Tutor API is running"}


@app.get("/signs")
async def get_signs():
    """Get list of all ASL signs."""
    return {"signs": ASL_CLASSES, "count": len(ASL_CLASSES)}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict ASL sign from image.
    
    Accepts base64 encoded image and returns prediction with confidence.
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)
        
        # Get prediction
        predictor = get_predictor()
        predicted_sign, confidence = predictor.predict_sign(image_array)
        
        # Get top-5 predictions
        top_5 = predictor.get_top_k(image_array, k=5)
        
        # Check if correct (if target provided)
        is_correct = None
        if request.target_sign:
            is_correct = predicted_sign.upper() == request.target_sign.upper()
        
        return PredictResponse(
            predicted_sign=predicted_sign,
            confidence=confidence,
            is_correct=is_correct,
            top_5=[{"sign": s, "confidence": c} for s, c in top_5]
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/next_sign", response_model=NextSignResponse)
async def get_next_sign(request: NextSignRequest):
    """
    Get the next sign to practice based on adaptive policy.
    
    Uses contextual bandit to select optimal next sign for learning.
    """
    try:
        # Get student model and bandit
        student = StudentModel(user_id=request.user_id, data_dir=DATA_DIR)
        bandit = get_bandit(request.user_id)
        
        # Select next sign
        sign = bandit.select_next_sign(student)
        
        # Get sign progress
        progress = student.get_progress(sign)
        
        return NextSignResponse(
            sign=sign,
            sign_image_url=f"/static/signs/{sign}.png",  # Frontend can use this
            mastery=progress.mastery,
            attempts=progress.attempts
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update", response_model=UpdateResponse)
async def update_progress(request: UpdateRequest):
    """
    Update student model and bandit after an attempt.
    
    Should be called after each practice attempt with the result.
    """
    try:
        # Get student model and bandit
        student = StudentModel(user_id=request.user_id, data_dir=DATA_DIR)
        bandit = get_bandit(request.user_id)
        
        # Get context before update
        context = student.get_context(request.sign_label)
        
        # Compute reward for bandit
        reward = compute_reward(request.correct, request.response_time)
        
        # Update student model
        progress = student.update(
            request.sign_label,
            request.correct,
            request.response_time
        )
        
        # Update bandit
        bandit.update(request.sign_label, context, reward)
        save_bandit(request.user_id)
        
        return UpdateResponse(
            success=True,
            new_mastery=progress.mastery,
            overall_mastery=student.get_overall_mastery()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/progress/{user_id}", response_model=ProgressResponse)
async def get_progress(user_id: str):
    """
    Get student's learning progress.
    
    Returns overall mastery, per-sign progress, and weak areas.
    """
    try:
        student = StudentModel(user_id=user_id, data_dir=DATA_DIR)
        
        # Get all progress
        all_progress = student.get_all_progress()
        progress_dict = {
            sign: {
                "mastery": p.mastery,
                "attempts": p.attempts,
                "correct": p.correct,
                "avg_time": p.avg_time,
                "streak": p.streak
            }
            for sign, p in all_progress.items()
        }
        
        # Count mastered signs
        mastered = sum(1 for p in all_progress.values() if p.mastery >= 0.8)
        
        return ProgressResponse(
            user_id=user_id,
            overall_mastery=student.get_overall_mastery(),
            total_attempts=student.data.total_attempts,
            total_sessions=student.data.total_sessions,
            signs_mastered=mastered,
            weak_signs=student.get_weak_signs(),
            progress_by_sign=progress_dict
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/start")
async def start_session(request: SessionRequest):
    """Start a new learning session."""
    try:
        student = StudentModel(user_id=request.user_id, data_dir=DATA_DIR)
        student.start_session()
        
        return {
            "success": True,
            "session_number": student.data.total_sessions,
            "mode": request.mode
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/session/reset")
async def reset_progress(request: NextSignRequest):
    """Reset student progress (use with caution!)."""
    try:
        student = StudentModel(user_id=request.user_id, data_dir=DATA_DIR)
        student.reset()
        
        # Reset bandit
        global _bandits
        if request.user_id in _bandits:
            del _bandits[request.user_id]
        
        return {"success": True, "message": "Progress reset"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/leaderboard")
async def get_leaderboard(limit: int = 10):
    """Get top learners by overall mastery."""
    try:
        users_dir = os.path.join(DATA_DIR)
        if not os.path.exists(users_dir):
            return {"leaderboard": []}
        
        leaderboard = []
        for filename in os.listdir(users_dir):
            if filename.endswith('.json') and not filename.endswith('_bandit.json'):
                user_id = filename.replace('.json', '')
                student = StudentModel(user_id=user_id, data_dir=DATA_DIR)
                leaderboard.append({
                    "user_id": user_id,
                    "overall_mastery": student.get_overall_mastery(),
                    "total_attempts": student.data.total_attempts,
                    "signs_mastered": sum(
                        1 for p in student.get_all_progress().values() 
                        if p.mastery >= 0.8
                    )
                })
        
        # Sort by mastery
        leaderboard.sort(key=lambda x: x["overall_mastery"], reverse=True)
        
        return {"leaderboard": leaderboard[:limit]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Run Server
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
