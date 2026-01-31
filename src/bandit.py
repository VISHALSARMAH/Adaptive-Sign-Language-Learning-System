"""
Contextual Bandit Policy Module
===============================
Implements adaptive sign selection using Thompson Sampling and LinUCB.
Chooses the optimal next sign to practice based on student context.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

from .dataset import ASL_CLASSES
from .student_model import StudentModel


class BanditPolicy(ABC):
    """Abstract base class for bandit policies."""
    
    @abstractmethod
    def select_next_sign(self, student_model: StudentModel) -> str:
        """Select the next sign to practice."""
        pass
    
    @abstractmethod
    def update(self, sign_label: str, context: np.ndarray, reward: float):
        """Update policy after observing reward."""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save policy parameters."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load policy parameters."""
        pass


class LinearThompsonSampling(BanditPolicy):
    """
    Linear Thompson Sampling Contextual Bandit.
    
    For each arm (sign), maintains Bayesian linear regression parameters
    to estimate expected reward given context features.
    
    Args:
        context_dim: Dimension of context feature vector
        arms: List of arm names (sign labels)
        lambda_prior: Regularization parameter / prior precision
        v_squared: Noise variance (R-squared in Thompson Sampling)
    """
    
    def __init__(
        self,
        context_dim: int = 6,
        arms: List[str] = ASL_CLASSES,
        lambda_prior: float = 1.0,
        v_squared: float = 0.25
    ):
        self.context_dim = context_dim
        self.arms = arms
        self.n_arms = len(arms)
        self.arm_to_idx = {arm: i for i, arm in enumerate(arms)}
        self.lambda_prior = lambda_prior
        self.v_squared = v_squared
        
        # Initialize parameters for each arm
        # B_a: precision matrix (inverse covariance)
        # mu_a: mean of weight distribution
        # f_a: accumulated rewards weighted by context
        self.B = [lambda_prior * np.eye(context_dim) for _ in range(self.n_arms)]
        self.mu = [np.zeros(context_dim) for _ in range(self.n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(self.n_arms)]
        
        # Track attempt counts per arm
        self.counts = np.zeros(self.n_arms)
    
    def select_next_sign(self, student_model: StudentModel) -> str:
        """
        Select next sign using Thompson Sampling.
        
        For each sign:
        1. Sample weight vector from posterior
        2. Compute expected reward using context
        3. Select sign with highest sampled reward
        
        Args:
            student_model: StudentModel instance with current learner state
        
        Returns:
            Sign label to practice next
        """
        sampled_rewards = []
        
        for i, sign in enumerate(self.arms):
            # Get context for this sign
            context = student_model.get_context(sign)
            
            # Sample from posterior
            B_inv = np.linalg.inv(self.B[i])
            theta_sample = np.random.multivariate_normal(
                self.mu[i], 
                self.v_squared * B_inv
            )
            
            # Compute sampled reward
            reward = np.dot(theta_sample, context)
            
            # Add exploration bonus for signs with few attempts
            # (prioritize signs the student hasn't practiced much)
            progress = student_model.get_progress(sign)
            
            # Reward shaping: prefer signs with low mastery
            mastery_bonus = 1.0 - progress.mastery
            
            # Spaced repetition: boost recently practiced but not mastered
            recency_bonus = 0.0
            if progress.last_attempt:
                from datetime import datetime
                last = datetime.fromisoformat(progress.last_attempt)
                hours_since = (datetime.now() - last).total_seconds() / 3600
                if hours_since < 24 and progress.mastery < 0.8:
                    recency_bonus = 0.1 * (24 - hours_since) / 24
            
            total_reward = reward + 0.5 * mastery_bonus + recency_bonus
            sampled_rewards.append(total_reward)
        
        # Select arm with highest sampled reward
        best_idx = np.argmax(sampled_rewards)
        return self.arms[best_idx]
    
    def update(self, sign_label: str, context: np.ndarray, reward: float):
        """
        Update posterior after observing reward.
        
        Uses Bayesian linear regression update:
        B_a = B_a + x @ x.T
        f_a = f_a + r * x
        mu_a = B_a^{-1} @ f_a
        
        Args:
            sign_label: The sign that was practiced
            context: Context feature vector used for selection
            reward: Observed reward (1 if correct & fast, 0 otherwise)
        """
        idx = self.arm_to_idx[sign_label]
        
        # Ensure context is column vector
        x = context.reshape(-1, 1)
        
        # Update precision matrix
        self.B[idx] = self.B[idx] + x @ x.T
        
        # Update accumulated reward
        self.f[idx] = self.f[idx] + reward * context
        
        # Update mean
        B_inv = np.linalg.inv(self.B[idx])
        self.mu[idx] = B_inv @ self.f[idx]
        
        # Update count
        self.counts[idx] += 1
    
    def save(self, path: str):
        """Save policy parameters to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'context_dim': self.context_dim,
            'arms': self.arms,
            'lambda_prior': self.lambda_prior,
            'v_squared': self.v_squared,
            'B': [b.tolist() for b in self.B],
            'mu': [m.tolist() for m in self.mu],
            'f': [f.tolist() for f in self.f],
            'counts': self.counts.tolist()
        }
        
        with open(path, 'w') as fp:
            json.dump(data, fp)
    
    def load(self, path: str):
        """Load policy parameters from file."""
        with open(path, 'r') as fp:
            data = json.load(fp)
        
        self.context_dim = data['context_dim']
        self.arms = data['arms']
        self.n_arms = len(self.arms)
        self.arm_to_idx = {arm: i for i, arm in enumerate(self.arms)}
        self.lambda_prior = data['lambda_prior']
        self.v_squared = data['v_squared']
        self.B = [np.array(b) for b in data['B']]
        self.mu = [np.array(m) for m in data['mu']]
        self.f = [np.array(f) for f in data['f']]
        self.counts = np.array(data['counts'])


class LinUCB(BanditPolicy):
    """
    Linear Upper Confidence Bound (LinUCB) Contextual Bandit.
    
    An alternative to Thompson Sampling that uses confidence bounds
    instead of sampling from the posterior.
    
    Args:
        context_dim: Dimension of context feature vector
        arms: List of arm names (sign labels)
        alpha: Exploration parameter (higher = more exploration)
    """
    
    def __init__(
        self,
        context_dim: int = 6,
        arms: List[str] = ASL_CLASSES,
        alpha: float = 1.0
    ):
        self.context_dim = context_dim
        self.arms = arms
        self.n_arms = len(arms)
        self.arm_to_idx = {arm: i for i, arm in enumerate(arms)}
        self.alpha = alpha
        
        # Initialize parameters
        self.A = [np.eye(context_dim) for _ in range(self.n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(self.n_arms)]
        self.counts = np.zeros(self.n_arms)
    
    def select_next_sign(self, student_model: StudentModel) -> str:
        """Select next sign using UCB exploration."""
        ucb_values = []
        
        for i, sign in enumerate(self.arms):
            context = student_model.get_context(sign)
            
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            
            # Predicted reward
            pred_reward = np.dot(theta, context)
            
            # Confidence bound
            confidence = self.alpha * np.sqrt(
                context @ A_inv @ context
            )
            
            # Mastery-based bonus
            progress = student_model.get_progress(sign)
            mastery_bonus = 1.0 - progress.mastery
            
            ucb = pred_reward + confidence + 0.3 * mastery_bonus
            ucb_values.append(ucb)
        
        best_idx = np.argmax(ucb_values)
        return self.arms[best_idx]
    
    def update(self, sign_label: str, context: np.ndarray, reward: float):
        """Update parameters after observing reward."""
        idx = self.arm_to_idx[sign_label]
        
        x = context.reshape(-1, 1)
        self.A[idx] = self.A[idx] + x @ x.T
        self.b[idx] = self.b[idx] + reward * context
        self.counts[idx] += 1
    
    def save(self, path: str):
        """Save policy parameters."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'context_dim': self.context_dim,
            'arms': self.arms,
            'alpha': self.alpha,
            'A': [a.tolist() for a in self.A],
            'b': [b.tolist() for b in self.b],
            'counts': self.counts.tolist()
        }
        
        with open(path, 'w') as fp:
            json.dump(data, fp)
    
    def load(self, path: str):
        """Load policy parameters."""
        with open(path, 'r') as fp:
            data = json.load(fp)
        
        self.context_dim = data['context_dim']
        self.arms = data['arms']
        self.n_arms = len(self.arms)
        self.arm_to_idx = {arm: i for i, arm in enumerate(self.arms)}
        self.alpha = data['alpha']
        self.A = [np.array(a) for a in data['A']]
        self.b = [np.array(b) for b in data['b']]
        self.counts = np.array(data['counts'])


class RandomPolicy(BanditPolicy):
    """Random baseline policy (for comparison)."""
    
    def __init__(self, arms: List[str] = ASL_CLASSES):
        self.arms = arms
    
    def select_next_sign(self, student_model: StudentModel) -> str:
        return np.random.choice(self.arms)
    
    def update(self, sign_label: str, context: np.ndarray, reward: float):
        pass  # No learning
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass


class FixedCurriculumPolicy(BanditPolicy):
    """Fixed curriculum baseline (A, B, C, ..., Z)."""
    
    def __init__(self, arms: List[str] = ASL_CLASSES, mastery_threshold: float = 0.8):
        self.arms = arms
        self.current_idx = 0
        self.mastery_threshold = mastery_threshold
    
    def select_next_sign(self, student_model: StudentModel) -> str:
        # Move to next sign if current is mastered
        while self.current_idx < len(self.arms):
            sign = self.arms[self.current_idx]
            if student_model.has_mastered(sign, self.mastery_threshold):
                self.current_idx += 1
            else:
                return sign
        
        # All signs mastered, cycle back
        self.current_idx = 0
        return self.arms[0]
    
    def update(self, sign_label: str, context: np.ndarray, reward: float):
        pass  # No learning
    
    def save(self, path: str):
        with open(path, 'w') as fp:
            json.dump({'current_idx': self.current_idx}, fp)
    
    def load(self, path: str):
        with open(path, 'r') as fp:
            data = json.load(fp)
        self.current_idx = data['current_idx']


def compute_reward(correct: bool, response_time: float, time_threshold: float = 3.0) -> float:
    """
    Compute reward for bandit update.
    
    Reward is 1 if:
    - Answer is correct AND
    - Response time is below threshold
    
    Otherwise, partial credit based on correctness and speed.
    
    Args:
        correct: Whether the attempt was correct
        response_time: Time taken to respond (seconds)
        time_threshold: Maximum time for full reward
    
    Returns:
        Reward value in [0, 1]
    """
    if not correct:
        return 0.0
    
    # Time-based reward decay
    if response_time <= time_threshold:
        time_bonus = 1.0
    else:
        # Decay reward for slow responses
        time_bonus = max(0.5, 1.0 - 0.1 * (response_time - time_threshold))
    
    return time_bonus


def get_bandit_policy(
    policy_type: str = 'thompson',
    **kwargs
) -> BanditPolicy:
    """
    Factory function to create bandit policies.
    
    Args:
        policy_type: 'thompson', 'linucb', 'random', or 'fixed'
        **kwargs: Additional arguments for the policy
    
    Returns:
        BanditPolicy instance
    """
    if policy_type == 'thompson':
        return LinearThompsonSampling(**kwargs)
    elif policy_type == 'linucb':
        return LinUCB(**kwargs)
    elif policy_type == 'random':
        return RandomPolicy(**kwargs)
    elif policy_type == 'fixed':
        return FixedCurriculumPolicy(**kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


# =============================================================================
# Simulation for offline evaluation
# =============================================================================

class SyntheticStudent:
    """
    Simulated student for offline bandit evaluation.
    
    Simulates learning dynamics with:
    - Different learning rates per sign
    - Forgetting over time
    - Variable response times
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        forgetting_rate: float = 0.01,
        noise_std: float = 0.1
    ):
        self.learning_rate = learning_rate
        self.forgetting_rate = forgetting_rate
        self.noise_std = noise_std
        
        # True skill levels (hidden from bandit)
        self.skill = {sign: 0.0 for sign in ASL_CLASSES}
        
        # Sign difficulties (some signs are harder)
        np.random.seed(42)
        self.difficulty = {
            sign: np.random.uniform(0.5, 1.5) 
            for sign in ASL_CLASSES
        }
    
    def attempt(self, sign: str) -> Tuple[bool, float]:
        """
        Simulate an attempt at a sign.
        
        Returns:
            (correct, response_time)
        """
        # Probability of success depends on skill and difficulty
        p_correct = self.skill[sign] / self.difficulty[sign]
        p_correct = np.clip(p_correct + np.random.normal(0, self.noise_std), 0, 1)
        
        correct = np.random.random() < p_correct
        
        # Response time depends on skill
        base_time = 5.0 - 3.0 * self.skill[sign]  # 2-5 seconds
        response_time = max(0.5, base_time + np.random.normal(0, 0.5))
        
        # Update skill (learning)
        if correct:
            self.skill[sign] = min(1.0, self.skill[sign] + self.learning_rate)
        else:
            self.skill[sign] = max(0.0, self.skill[sign] + 0.02)
        
        return correct, response_time
    
    def forget(self):
        """Apply forgetting to all skills."""
        for sign in self.skill:
            self.skill[sign] *= (1 - self.forgetting_rate)


def run_simulation(
    policy: BanditPolicy,
    n_steps: int = 500,
    student_lr: float = 0.1,
    mastery_threshold: float = 0.8
) -> Dict:
    """
    Run offline simulation comparing bandit policies.
    
    Args:
        policy: Bandit policy to evaluate
        n_steps: Number of practice steps
        student_lr: Synthetic student learning rate
        mastery_threshold: Mastery threshold
    
    Returns:
        Dictionary with simulation results
    """
    # Create synthetic student and student model
    synth_student = SyntheticStudent(learning_rate=student_lr)
    student_model = StudentModel(user_id='simulation', data_dir='/tmp/sim_users')
    
    # Track metrics
    cumulative_correct = 0
    correct_history = []
    mastery_history = []
    signs_mastered_history = []
    
    for step in range(n_steps):
        # Select next sign
        sign = policy.select_next_sign(student_model)
        
        # Get context before attempt
        context = student_model.get_context(sign)
        
        # Simulate attempt
        correct, response_time = synth_student.attempt(sign)
        
        # Compute reward for bandit
        reward = compute_reward(correct, response_time)
        
        # Update student model
        student_model.update(sign, correct, response_time)
        
        # Update bandit policy
        policy.update(sign, context, reward)
        
        # Track metrics
        cumulative_correct += int(correct)
        correct_history.append(cumulative_correct / (step + 1))
        mastery_history.append(student_model.get_overall_mastery())
        
        # Count mastered signs
        n_mastered = sum(
            1 for s in ASL_CLASSES 
            if student_model.has_mastered(s, mastery_threshold)
        )
        signs_mastered_history.append(n_mastered)
        
        # Apply forgetting periodically
        if step % 10 == 0:
            synth_student.forget()
    
    return {
        'accuracy': correct_history,
        'mastery': mastery_history,
        'signs_mastered': signs_mastered_history,
        'final_accuracy': correct_history[-1],
        'final_mastery': mastery_history[-1],
        'final_signs_mastered': signs_mastered_history[-1]
    }


def compare_policies(n_steps: int = 500, n_runs: int = 5):
    """
    Compare different bandit policies.
    
    Args:
        n_steps: Steps per simulation
        n_runs: Number of runs to average
    """
    import matplotlib.pyplot as plt
    
    policies = {
        'Thompson Sampling': lambda: LinearThompsonSampling(),
        'LinUCB': lambda: LinUCB(),
        'Random': lambda: RandomPolicy(),
        'Fixed Curriculum': lambda: FixedCurriculumPolicy()
    }
    
    results = {name: {'accuracy': [], 'mastery': [], 'signs_mastered': []} 
               for name in policies}
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        
        for name, policy_fn in policies.items():
            policy = policy_fn()
            run_result = run_simulation(policy, n_steps=n_steps)
            
            results[name]['accuracy'].append(run_result['accuracy'])
            results[name]['mastery'].append(run_result['mastery'])
            results[name]['signs_mastered'].append(run_result['signs_mastered'])
    
    # Average results
    avg_results = {}
    for name, data in results.items():
        avg_results[name] = {
            'accuracy': np.mean(data['accuracy'], axis=0),
            'mastery': np.mean(data['mastery'], axis=0),
            'signs_mastered': np.mean(data['signs_mastered'], axis=0)
        }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    steps = range(1, n_steps + 1)
    
    for name, data in avg_results.items():
        axes[0].plot(steps, data['accuracy'], label=name)
        axes[1].plot(steps, data['mastery'], label=name)
        axes[2].plot(steps, data['signs_mastered'], label=name)
    
    axes[0].set_xlabel('Practice Steps')
    axes[0].set_ylabel('Cumulative Accuracy')
    axes[0].set_title('Learning Efficiency')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Practice Steps')
    axes[1].set_ylabel('Overall Mastery')
    axes[1].set_title('Mastery Progress')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].set_xlabel('Practice Steps')
    axes[2].set_ylabel('Signs Mastered')
    axes[2].set_title('Signs Mastered Over Time')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('bandit_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return avg_results


if __name__ == "__main__":
    print("Testing Contextual Bandit...")
    
    # Test Thompson Sampling
    policy = LinearThompsonSampling()
    student = StudentModel(user_id='bandit_test', data_dir='/tmp/test_users')
    
    print("\nSimulating 20 practice steps...")
    for i in range(20):
        sign = policy.select_next_sign(student)
        context = student.get_context(sign)
        
        # Simulate outcome
        correct = np.random.random() > 0.4
        response_time = np.random.uniform(1, 5)
        reward = compute_reward(correct, response_time)
        
        student.update(sign, correct, response_time)
        policy.update(sign, context, reward)
        
        print(f"Step {i+1}: Practice '{sign}', "
              f"Correct: {correct}, Reward: {reward:.2f}")
    
    print(f"\nFinal overall mastery: {student.get_overall_mastery():.2%}")
