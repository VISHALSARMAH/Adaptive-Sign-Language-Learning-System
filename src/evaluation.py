"""
Evaluation Helpers Module
=========================
Research helpers for logging, A/B testing, and analysis.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset import ASL_CLASSES
from .student_model import StudentModel
from .bandit import (
    LinearThompsonSampling, 
    LinUCB, 
    RandomPolicy, 
    FixedCurriculumPolicy,
    compute_reward,
    run_simulation
)


# =============================================================================
# Session Logging
# =============================================================================

@dataclass
class AttemptLog:
    """Log entry for a single practice attempt."""
    timestamp: str
    sign: str
    correct: bool
    response_time: float
    confidence: float
    mastery_before: float
    mastery_after: float


@dataclass
class SessionLog:
    """Log for a complete learning session."""
    session_id: str
    user_id: str
    mode: str  # 'adaptive' or 'fixed'
    start_time: str
    end_time: Optional[str] = None
    attempts: List[dict] = field(default_factory=list)
    signs_practiced: Dict[str, int] = field(default_factory=dict)
    accuracy_by_sign: Dict[str, float] = field(default_factory=dict)


class SessionLogger:
    """
    Logs per-session data for research analysis.
    
    Tracks:
    - Number of attempts per sign
    - Accuracy per sign
    - Time-to-mastery
    - Response times
    """
    
    def __init__(self, log_dir: str = 'data/logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[SessionLog] = None
    
    def start_session(self, user_id: str, mode: str = 'adaptive') -> str:
        """Start a new session and return session ID."""
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session = SessionLog(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            start_time=datetime.now().isoformat()
        )
        
        return session_id
    
    def log_attempt(
        self,
        sign: str,
        correct: bool,
        response_time: float,
        confidence: float,
        mastery_before: float,
        mastery_after: float
    ):
        """Log a single practice attempt."""
        if self.current_session is None:
            raise ValueError("No active session. Call start_session first.")
        
        attempt = AttemptLog(
            timestamp=datetime.now().isoformat(),
            sign=sign,
            correct=correct,
            response_time=response_time,
            confidence=confidence,
            mastery_before=mastery_before,
            mastery_after=mastery_after
        )
        
        self.current_session.attempts.append(asdict(attempt))
        
        # Update sign counts
        if sign not in self.current_session.signs_practiced:
            self.current_session.signs_practiced[sign] = 0
        self.current_session.signs_practiced[sign] += 1
    
    def end_session(self):
        """End the current session and save to file."""
        if self.current_session is None:
            return
        
        self.current_session.end_time = datetime.now().isoformat()
        
        # Calculate accuracy by sign
        sign_correct = {}
        sign_total = {}
        
        for attempt in self.current_session.attempts:
            sign = attempt['sign']
            if sign not in sign_correct:
                sign_correct[sign] = 0
                sign_total[sign] = 0
            sign_total[sign] += 1
            if attempt['correct']:
                sign_correct[sign] += 1
        
        self.current_session.accuracy_by_sign = {
            sign: sign_correct.get(sign, 0) / sign_total[sign]
            for sign in sign_total
        }
        
        # Save to file
        filename = f"{self.current_session.session_id}.json"
        filepath = self.log_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self.current_session), f, indent=2)
        
        print(f"Session saved to {filepath}")
        
        session = self.current_session
        self.current_session = None
        
        return session
    
    def load_session(self, session_id: str) -> SessionLog:
        """Load a session from file."""
        filepath = self.log_dir / f"{session_id}.json"
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return SessionLog(**data)
    
    def load_all_sessions(self, user_id: Optional[str] = None) -> List[SessionLog]:
        """Load all sessions, optionally filtered by user."""
        sessions = []
        
        for filepath in self.log_dir.glob('*.json'):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if user_id is None or data['user_id'] == user_id:
                sessions.append(SessionLog(**data))
        
        return sessions


# =============================================================================
# A/B Testing
# =============================================================================

class ABTestManager:
    """
    Manages A/B testing between adaptive and fixed curriculum modes.
    
    Mode A: Fixed curriculum (A, B, C, ..., Z)
    Mode B: Adaptive bandit curriculum
    """
    
    def __init__(
        self,
        data_dir: str = 'data/ab_test',
        mastery_threshold: float = 0.8
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mastery_threshold = mastery_threshold
        
        # Track user assignments
        self.assignments_file = self.data_dir / 'assignments.json'
        self.assignments = self._load_assignments()
    
    def _load_assignments(self) -> Dict[str, str]:
        """Load user-to-mode assignments."""
        if self.assignments_file.exists():
            with open(self.assignments_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_assignments(self):
        """Save user-to-mode assignments."""
        with open(self.assignments_file, 'w') as f:
            json.dump(self.assignments, f, indent=2)
    
    def assign_user(self, user_id: str, force_mode: Optional[str] = None) -> str:
        """
        Assign a user to a test mode (random if not specified).
        
        Args:
            user_id: User identifier
            force_mode: Force assignment to 'fixed' or 'adaptive'
        
        Returns:
            Assigned mode
        """
        if user_id in self.assignments:
            return self.assignments[user_id]
        
        if force_mode:
            mode = force_mode
        else:
            # Random assignment with 50/50 split
            mode = np.random.choice(['fixed', 'adaptive'])
        
        self.assignments[user_id] = mode
        self._save_assignments()
        
        return mode
    
    def get_user_mode(self, user_id: str) -> str:
        """Get the mode assigned to a user."""
        if user_id not in self.assignments:
            return self.assign_user(user_id)
        return self.assignments[user_id]
    
    def get_policy(self, user_id: str):
        """Get the appropriate policy for a user."""
        mode = self.get_user_mode(user_id)
        
        if mode == 'fixed':
            return FixedCurriculumPolicy(mastery_threshold=self.mastery_threshold)
        else:
            return LinearThompsonSampling()
    
    def run_ab_experiment(
        self,
        n_users_per_group: int = 10,
        n_steps_per_user: int = 200,
        student_lr: float = 0.1
    ) -> Dict:
        """
        Run an A/B experiment with synthetic students.
        
        Args:
            n_users_per_group: Number of users per condition
            n_steps_per_user: Practice steps per user
            student_lr: Synthetic student learning rate
        
        Returns:
            Experiment results dictionary
        """
        from .bandit import SyntheticStudent
        
        results = {
            'fixed': {'users': [], 'final_mastery': [], 'attempts_to_mastery': []},
            'adaptive': {'users': [], 'final_mastery': [], 'attempts_to_mastery': []}
        }
        
        for mode in ['fixed', 'adaptive']:
            print(f"\nRunning {mode} condition...")
            
            for user_idx in range(n_users_per_group):
                user_id = f"{mode}_{user_idx}"
                
                # Create policy
                if mode == 'fixed':
                    policy = FixedCurriculumPolicy()
                else:
                    policy = LinearThompsonSampling()
                
                # Create synthetic student
                synth_student = SyntheticStudent(learning_rate=student_lr)
                student_model = StudentModel(
                    user_id=user_id,
                    data_dir=str(self.data_dir / 'sim_users')
                )
                
                # Track when signs are mastered
                mastery_times = {}
                
                for step in range(n_steps_per_user):
                    sign = policy.select_next_sign(student_model)
                    context = student_model.get_context(sign)
                    correct, response_time = synth_student.attempt(sign)
                    reward = compute_reward(correct, response_time)
                    
                    student_model.update(sign, correct, response_time)
                    policy.update(sign, context, reward)
                    
                    # Check if sign was just mastered
                    if sign not in mastery_times:
                        if student_model.has_mastered(sign, self.mastery_threshold):
                            mastery_times[sign] = step
                    
                    # Apply forgetting
                    if step % 10 == 0:
                        synth_student.forget()
                
                # Record results
                results[mode]['users'].append(user_id)
                results[mode]['final_mastery'].append(student_model.get_overall_mastery())
                
                # Average attempts to mastery
                if mastery_times:
                    avg_atm = np.mean(list(mastery_times.values()))
                    results[mode]['attempts_to_mastery'].append(avg_atm)
                else:
                    results[mode]['attempts_to_mastery'].append(n_steps_per_user)
                
                print(f"  User {user_idx + 1}: mastery={student_model.get_overall_mastery():.2%}, "
                      f"signs_mastered={len(mastery_times)}")
        
        return results
    
    def analyze_results(self, results: Dict) -> pd.DataFrame:
        """
        Analyze A/B test results.
        
        Returns a DataFrame with summary statistics.
        """
        data = []
        
        for mode in ['fixed', 'adaptive']:
            for i, user in enumerate(results[mode]['users']):
                data.append({
                    'user_id': user,
                    'mode': mode,
                    'final_mastery': results[mode]['final_mastery'][i],
                    'attempts_to_mastery': results[mode]['attempts_to_mastery'][i]
                })
        
        df = pd.DataFrame(data)
        
        # Summary statistics
        summary = df.groupby('mode').agg({
            'final_mastery': ['mean', 'std'],
            'attempts_to_mastery': ['mean', 'std']
        }).round(4)
        
        print("\n=== A/B Test Results ===")
        print(summary)
        
        # Statistical significance (t-test)
        from scipy import stats
        
        fixed_mastery = df[df['mode'] == 'fixed']['final_mastery']
        adaptive_mastery = df[df['mode'] == 'adaptive']['final_mastery']
        
        t_stat, p_value = stats.ttest_ind(fixed_mastery, adaptive_mastery)
        print(f"\nMastery t-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        if p_value < 0.05:
            winner = 'adaptive' if adaptive_mastery.mean() > fixed_mastery.mean() else 'fixed'
            print(f"Significant difference! {winner.upper()} mode is better.")
        else:
            print("No significant difference between modes.")
        
        return df
    
    def plot_results(self, results: Dict, save_path: Optional[str] = None):
        """Plot A/B test results."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Final Mastery
        modes = ['fixed', 'adaptive']
        mastery_data = [results[m]['final_mastery'] for m in modes]
        
        axes[0].boxplot(mastery_data, labels=['Fixed', 'Adaptive'])
        axes[0].set_ylabel('Final Mastery')
        axes[0].set_title('Final Mastery by Condition')
        axes[0].grid(True, alpha=0.3)
        
        # Attempts to Mastery
        atm_data = [results[m]['attempts_to_mastery'] for m in modes]
        
        axes[1].boxplot(atm_data, labels=['Fixed', 'Adaptive'])
        axes[1].set_ylabel('Attempts to Mastery')
        axes[1].set_title('Learning Efficiency')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_learning_curve(
    session_logs: List[SessionLog],
    window_size: int = 10
) -> pd.DataFrame:
    """
    Analyze learning curves from session logs.
    
    Args:
        session_logs: List of session logs
        window_size: Rolling window for smoothing
    
    Returns:
        DataFrame with learning curve data
    """
    all_attempts = []
    
    for session in session_logs:
        for i, attempt in enumerate(session.attempts):
            all_attempts.append({
                'session_id': session.session_id,
                'user_id': session.user_id,
                'mode': session.mode,
                'attempt_num': i + 1,
                'correct': attempt['correct'],
                'response_time': attempt['response_time'],
                'mastery': attempt['mastery_after']
            })
    
    df = pd.DataFrame(all_attempts)
    
    # Calculate rolling accuracy
    df['rolling_accuracy'] = df.groupby('user_id')['correct'].transform(
        lambda x: x.rolling(window=window_size, min_periods=1).mean()
    )
    
    return df


def plot_learning_curves_comparison(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot learning curves comparing adaptive vs fixed curriculum."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy over time
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        
        # Average across users
        avg_accuracy = mode_df.groupby('attempt_num')['rolling_accuracy'].mean()
        
        axes[0].plot(avg_accuracy.index, avg_accuracy.values, 
                    label=mode.capitalize(), linewidth=2)
    
    axes[0].set_xlabel('Attempt Number')
    axes[0].set_ylabel('Rolling Accuracy')
    axes[0].set_title('Learning Curve: Accuracy Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Mastery over time
    for mode in df['mode'].unique():
        mode_df = df[df['mode'] == mode]
        avg_mastery = mode_df.groupby('attempt_num')['mastery'].mean()
        
        axes[1].plot(avg_mastery.index, avg_mastery.values,
                    label=mode.capitalize(), linewidth=2)
    
    axes[1].set_xlabel('Attempt Number')
    axes[1].set_ylabel('Overall Mastery')
    axes[1].set_title('Mastery Progress Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_learner_summary(user_id: str, data_dir: str = 'data/users') -> Dict:
    """
    Generate a summary report for a learner.
    
    Returns:
        Dictionary with learner statistics
    """
    student = StudentModel(user_id=user_id, data_dir=data_dir)
    
    all_progress = student.get_all_progress()
    
    summary = {
        'user_id': user_id,
        'overall_mastery': student.get_overall_mastery(),
        'total_attempts': student.data.total_attempts,
        'total_sessions': student.data.total_sessions,
        'total_time_spent_minutes': student.data.total_time_spent / 60,
        'signs_mastered': len(student.get_strong_signs(0.8)),
        'signs_learning': len([p for p in all_progress.values() if 0.3 <= p.mastery < 0.8]),
        'signs_not_started': len([p for p in all_progress.values() if p.attempts == 0]),
        'weak_signs': student.get_weak_signs(0.5, 5),
        'strongest_signs': [
            sign for sign, p in sorted(
                all_progress.items(), 
                key=lambda x: x[1].mastery, 
                reverse=True
            )[:5]
        ],
        'avg_response_time': np.mean([p.avg_time for p in all_progress.values()]),
        'accuracy': sum(p.correct for p in all_progress.values()) / 
                   max(1, sum(p.attempts for p in all_progress.values()))
    }
    
    return summary


def print_learner_report(user_id: str, data_dir: str = 'data/users'):
    """Print a formatted learner report."""
    summary = generate_learner_summary(user_id, data_dir)
    
    print("=" * 50)
    print(f"LEARNER REPORT: {summary['user_id']}")
    print("=" * 50)
    print(f"Overall Mastery: {summary['overall_mastery']:.1%}")
    print(f"Total Attempts: {summary['total_attempts']}")
    print(f"Total Sessions: {summary['total_sessions']}")
    print(f"Time Spent: {summary['total_time_spent_minutes']:.1f} minutes")
    print(f"Accuracy: {summary['accuracy']:.1%}")
    print(f"Avg Response Time: {summary['avg_response_time']:.2f}s")
    print("-" * 50)
    print(f"Signs Mastered: {summary['signs_mastered']}/29")
    print(f"Signs Learning: {summary['signs_learning']}")
    print(f"Signs Not Started: {summary['signs_not_started']}")
    print("-" * 50)
    print(f"Strongest Signs: {', '.join(summary['strongest_signs'])}")
    print(f"Weak Signs (need practice): {', '.join(summary['weak_signs'])}")
    print("=" * 50)


if __name__ == "__main__":
    # Run A/B experiment demo
    print("Running A/B Experiment Demo...")
    
    ab_manager = ABTestManager(data_dir='data/ab_test_demo')
    
    results = ab_manager.run_ab_experiment(
        n_users_per_group=5,
        n_steps_per_user=100,
        student_lr=0.12
    )
    
    df = ab_manager.analyze_results(results)
    ab_manager.plot_results(results, save_path='ab_test_results.png')
