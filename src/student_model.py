"""
Student Model Module
====================
Tracks per-sign mastery for each learner using exponential moving averages.
Provides context features for the adaptive bandit policy.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import numpy as np

from .dataset import ASL_CLASSES


@dataclass
class SignProgress:
    """Progress tracking for a single sign."""
    mastery: float = 0.0          # Mastery level [0, 1]
    attempts: int = 0             # Total attempts
    correct: int = 0              # Total correct attempts
    avg_time: float = 3.0         # Average response time (seconds)
    last_attempt: Optional[str] = None  # ISO timestamp of last attempt
    streak: int = 0               # Current correct streak
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SignProgress':
        return cls(**data)


@dataclass
class StudentData:
    """Complete student data structure."""
    user_id: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_sessions: int = 0
    total_attempts: int = 0
    total_time_spent: float = 0.0  # seconds
    signs: Dict[str, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'StudentData':
        return cls(**data)


class StudentModel:
    """
    Student mastery tracking model.
    
    Maintains per-sign mastery for each learner with:
    - mastery ∈ [0, 1]
    - attempts count
    - average response time
    - streak tracking
    
    Args:
        user_id: Unique user identifier
        data_dir: Directory to store user data
        alpha: Exponential moving average decay for mastery updates
        time_alpha: EMA decay for response time updates
    """
    
    def __init__(
        self,
        user_id: str,
        data_dir: str = 'data/users',
        alpha: float = 0.25,
        time_alpha: float = 0.3
    ):
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        self.alpha = alpha
        self.time_alpha = time_alpha
        self.classes = ASL_CLASSES
        
        # Create data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize student data
        self.data = self.load(user_id)
    
    @property
    def file_path(self) -> Path:
        """Path to user's JSON file."""
        return self.data_dir / f"{self.user_id}.json"
    
    def load(self, user_id: str) -> StudentData:
        """
        Load student data from file or create new.
        
        Args:
            user_id: User identifier
        
        Returns:
            StudentData instance
        """
        if self.file_path.exists():
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            return StudentData.from_dict(data)
        else:
            # Initialize new student
            return self._initialize_new_student(user_id)
    
    def _initialize_new_student(self, user_id: str) -> StudentData:
        """Create a new student with default values."""
        signs = {}
        for sign in self.classes:
            signs[sign] = SignProgress().to_dict()
        
        data = StudentData(
            user_id=user_id,
            signs=signs
        )
        return data
    
    def save(self, user_id: Optional[str] = None):
        """
        Save student data to file.
        
        Args:
            user_id: Optional user ID (uses self.user_id if None)
        """
        if user_id and user_id != self.user_id:
            # Save to different user
            path = self.data_dir / f"{user_id}.json"
        else:
            path = self.file_path
        
        with open(path, 'w') as f:
            json.dump(self.data.to_dict(), f, indent=2)
    
    def update(
        self,
        sign_label: str,
        correct: bool,
        response_time: float
    ) -> SignProgress:
        """
        Update mastery for a sign after an attempt.
        
        Args:
            sign_label: The sign that was practiced
            correct: Whether the attempt was correct
            response_time: Time taken to respond (seconds)
        
        Returns:
            Updated SignProgress for the sign
        """
        if sign_label not in self.data.signs:
            self.data.signs[sign_label] = SignProgress().to_dict()
        
        progress = SignProgress.from_dict(self.data.signs[sign_label])
        
        # Update attempts
        progress.attempts += 1
        if correct:
            progress.correct += 1
            progress.streak += 1
        else:
            progress.streak = 0
        
        # Update mastery with exponential moving average
        correct_val = 1.0 if correct else 0.0
        progress.mastery = self.alpha * correct_val + (1 - self.alpha) * progress.mastery
        
        # Clamp mastery to [0, 1]
        progress.mastery = max(0.0, min(1.0, progress.mastery))
        
        # Update average response time (EMA)
        progress.avg_time = (
            self.time_alpha * response_time + 
            (1 - self.time_alpha) * progress.avg_time
        )
        
        # Update timestamp
        progress.last_attempt = datetime.now().isoformat()
        
        # Update global stats
        self.data.total_attempts += 1
        self.data.total_time_spent += response_time
        
        # Save back
        self.data.signs[sign_label] = progress.to_dict()
        self.save()
        
        return progress
    
    def get_progress(self, sign_label: str) -> SignProgress:
        """Get progress for a specific sign."""
        if sign_label not in self.data.signs:
            return SignProgress()
        return SignProgress.from_dict(self.data.signs[sign_label])
    
    def get_all_progress(self) -> Dict[str, SignProgress]:
        """Get progress for all signs."""
        return {
            sign: SignProgress.from_dict(data) 
            for sign, data in self.data.signs.items()
        }
    
    def get_context(self, sign_label: str) -> np.ndarray:
        """
        Get context feature vector for a sign.
        
        Features:
        [0] mastery - current mastery level
        [1] attempts_normalized - normalized attempt count
        [2] avg_time_normalized - normalized average response time
        [3] days_since_last - days since last attempt (normalized)
        [4] learner_avg_mastery - average mastery across all signs
        [5] streak_normalized - current streak (normalized)
        
        Args:
            sign_label: The sign to get context for
        
        Returns:
            Feature vector of shape (6,)
        """
        progress = self.get_progress(sign_label)
        
        # Feature 1: Mastery
        mastery = progress.mastery
        
        # Feature 2: Normalized attempts (sigmoid scaling)
        attempts_norm = 1 / (1 + np.exp(-0.1 * (progress.attempts - 20)))
        
        # Feature 3: Normalized response time (assuming 1-10s range)
        avg_time_norm = min(progress.avg_time / 10.0, 1.0)
        
        # Feature 4: Days since last attempt
        if progress.last_attempt:
            last_dt = datetime.fromisoformat(progress.last_attempt)
            days_since = (datetime.now() - last_dt).total_seconds() / 86400
            days_since_norm = min(days_since / 30.0, 1.0)  # Cap at 30 days
        else:
            days_since_norm = 1.0  # Never attempted
        
        # Feature 5: Average mastery across all signs
        all_mastery = [
            SignProgress.from_dict(d).mastery 
            for d in self.data.signs.values()
        ]
        avg_mastery = np.mean(all_mastery) if all_mastery else 0.0
        
        # Feature 6: Normalized streak
        streak_norm = min(progress.streak / 10.0, 1.0)
        
        return np.array([
            mastery,
            attempts_norm,
            avg_time_norm,
            days_since_norm,
            avg_mastery,
            streak_norm
        ], dtype=np.float32)
    
    def get_mastery_summary(self) -> Dict[str, float]:
        """Get mastery levels for all signs."""
        return {
            sign: SignProgress.from_dict(data).mastery
            for sign, data in self.data.signs.items()
        }
    
    def get_weak_signs(self, threshold: float = 0.5, top_k: int = 5) -> List[str]:
        """Get signs with lowest mastery."""
        mastery = self.get_mastery_summary()
        sorted_signs = sorted(mastery.items(), key=lambda x: x[1])
        return [sign for sign, m in sorted_signs[:top_k] if m < threshold]
    
    def get_strong_signs(self, threshold: float = 0.8) -> List[str]:
        """Get signs with high mastery."""
        mastery = self.get_mastery_summary()
        return [sign for sign, m in mastery.items() if m >= threshold]
    
    def has_mastered(self, sign_label: str, threshold: float = 0.8) -> bool:
        """Check if a sign has been mastered."""
        return self.get_progress(sign_label).mastery >= threshold
    
    def get_overall_mastery(self) -> float:
        """Get average mastery across all signs."""
        mastery = self.get_mastery_summary()
        return np.mean(list(mastery.values()))
    
    def start_session(self):
        """Mark the start of a new learning session."""
        self.data.total_sessions += 1
        self.save()
    
    def reset(self):
        """Reset all progress (use with caution!)."""
        self.data = self._initialize_new_student(self.user_id)
        self.save()
    
    def __repr__(self) -> str:
        overall = self.get_overall_mastery()
        return (
            f"StudentModel(user_id='{self.user_id}', "
            f"overall_mastery={overall:.2%}, "
            f"attempts={self.data.total_attempts})"
        )


def get_student_model(user_id: str, data_dir: str = 'data/users') -> StudentModel:
    """Convenience function to get a StudentModel instance."""
    return StudentModel(user_id=user_id, data_dir=data_dir)


if __name__ == "__main__":
    # Test the student model
    print("Testing StudentModel...")
    
    # Create a test student
    student = StudentModel(user_id='test_user')
    print(f"Created: {student}")
    
    # Simulate some learning
    import random
    for _ in range(10):
        sign = random.choice(ASL_CLASSES[:5])  # Only first 5 signs
        correct = random.random() > 0.3
        time_taken = random.uniform(1.0, 5.0)
        student.update(sign, correct, time_taken)
    
    print(f"\nAfter 10 attempts: {student}")
    print(f"Weak signs: {student.get_weak_signs()}")
    print(f"Strong signs: {student.get_strong_signs()}")
    
    # Get context for a sign
    context = student.get_context('A')
    print(f"\nContext for 'A': {context}")
