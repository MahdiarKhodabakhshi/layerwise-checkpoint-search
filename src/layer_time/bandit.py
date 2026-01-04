"""Linear contextual bandit algorithms for checkpoint-layer selection.

Step 5 of the workflow: Spend a limited evaluation budget using a structured bandit.

Implements linUCB (Linear Upper Confidence Bound) for selecting which
(checkpoint, layer) pairs to evaluate on downstream tasks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BanditState:
    """State of the bandit algorithm."""
    
    # Context dimension (number of features)
    context_dim: int
    
    # LinUCB parameters
    alpha: float = 1.0  # Exploration parameter
    
    # Per-arm statistics (for disjoint linear bandits)
    A: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)  # A_a matrices
    b: Dict[Tuple[str, int], np.ndarray] = field(default_factory=dict)  # b_a vectors
    
    # Track which arms have been pulled
    arm_counts: Dict[Tuple[str, int], int] = field(default_factory=dict)
    arm_rewards: Dict[Tuple[str, int], List[float]] = field(default_factory=dict)
    
    # Trajectory (history of selections)
    trajectory: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize data structures."""
        if self.context_dim <= 0:
            raise ValueError("context_dim must be > 0")


class LinUCB:
    """
    Linear Upper Confidence Bound (LinUCB) contextual bandit.
    
    Selects arms (checkpoint, layer pairs) based on context features (representation metrics)
    using the LinUCB algorithm.
    """
    
    def __init__(
        self,
        context_dim: int,
        alpha: float = 1.0,
        state: Optional[BanditState] = None,
    ):
        """
        Initialize LinUCB bandit.
        
        Args:
            context_dim: Dimension of context feature vector
            alpha: Exploration parameter (higher = more exploration)
            state: Optional initial state (for resuming)
        """
        if state is not None:
            self.state = state
        else:
            self.state = BanditState(context_dim=context_dim, alpha=alpha)
    
    def select_arm(
        self,
        available_arms: List[Tuple[str, int]],
        context_features: Dict[Tuple[str, int], np.ndarray],
    ) -> Tuple[str, int]:
        """
        Select an arm (checkpoint, layer) to pull based on context.
        
        Args:
            available_arms: List of (checkpoint, layer) tuples to choose from
            context_features: Dict mapping (checkpoint, layer) -> context feature vector
        
        Returns:
            Selected (checkpoint, layer) tuple
        """
        if not available_arms:
            raise ValueError("No arms available")
        
        scores: Dict[Tuple[str, int], float] = {}
        
        for arm in available_arms:
            if arm not in context_features:
                # Missing context - use high score to encourage exploration
                scores[arm] = 1e6
                continue
            
            x = context_features[arm]
            if len(x) != self.state.context_dim:
                raise ValueError(
                    f"Context dimension mismatch: expected {self.state.context_dim}, got {len(x)}"
                )
            
            x = x.reshape(-1, 1)  # Column vector
            
            # Initialize A and b if needed
            if arm not in self.state.A:
                self.state.A[arm] = np.eye(self.state.context_dim)
                self.state.b[arm] = np.zeros((self.state.context_dim, 1))
            
            A_a = self.state.A[arm]
            b_a = self.state.b[arm]
            
            # Compute A_a inverse
            try:
                A_inv = np.linalg.inv(A_a)
            except np.linalg.LinAlgError:
                # If singular, use pseudo-inverse
                A_inv = np.linalg.pinv(A_a)
            
            # Compute mean reward estimate: theta_a = A_a^{-1} * b_a
            theta_a = A_inv @ b_a
            
            # Compute UCB score: x^T * theta_a + alpha * sqrt(x^T * A_a^{-1} * x)
            mean_reward = (x.T @ theta_a)[0, 0]
            uncertainty = self.state.alpha * np.sqrt((x.T @ A_inv @ x)[0, 0])
            ucb_score = mean_reward + uncertainty
            
            scores[arm] = ucb_score
        
        # Select arm with highest UCB score
        selected_arm = max(available_arms, key=lambda a: scores.get(a, -1e6))
        
        return selected_arm
    
    def update(
        self,
        arm: Tuple[str, int],
        reward: float,
        context: np.ndarray,
    ) -> None:
        """
        Update bandit state after observing a reward.
        
        Args:
            arm: The (checkpoint, layer) arm that was pulled
            reward: Observed reward (z-scored)
            context: Context feature vector used for selection
        """
        if len(context) != self.state.context_dim:
            raise ValueError(
                f"Context dimension mismatch: expected {self.state.context_dim}, got {len(context)}"
            )
        
        x = context.reshape(-1, 1)  # Column vector
        
        # Initialize if needed
        if arm not in self.state.A:
            self.state.A[arm] = np.eye(self.state.context_dim)
            self.state.b[arm] = np.zeros((self.state.context_dim, 1))
        
        if arm not in self.state.arm_counts:
            self.state.arm_counts[arm] = 0
        if arm not in self.state.arm_rewards:
            self.state.arm_rewards[arm] = []
        
        # Update A_a and b_a: A_a = A_a + x * x^T, b_a = b_a + reward * x
        self.state.A[arm] += x @ x.T
        self.state.b[arm] += reward * x
        
        # Update counts and rewards
        self.state.arm_counts[arm] = self.state.arm_counts.get(arm, 0) + 1
        self.state.arm_rewards[arm].append(reward)
        
        # Record in trajectory
        self.state.trajectory.append({
            "arm": arm,
            "reward": reward,
            "context": context.tolist(),
            "count": self.state.arm_counts[arm],
        })
    
    def get_trajectory(self) -> List[Dict]:
        """Get the history of arm selections and rewards."""
        return self.state.trajectory.copy()
    
    def get_best_arm(self) -> Optional[Tuple[str, int]]:
        """Get the arm with the highest average reward."""
        if not self.state.arm_rewards:
            return None
        
        avg_rewards = {
            arm: np.mean(rewards)
            for arm, rewards in self.state.arm_rewards.items()
        }
        
        return max(avg_rewards.items(), key=lambda x: x[1])[0]
    
    def save_state(self, path: Path) -> None:
        """Save bandit state to disk."""
        state_dict = {
            "context_dim": self.state.context_dim,
            "alpha": self.state.alpha,
            "A": {f"{c}@{l}": A.tolist() for (c, l), A in self.state.A.items()},
            "b": {f"{c}@{l}": b.tolist() for (c, l), b in self.state.b.items()},
            "arm_counts": {f"{c}@{l}": count for (c, l), count in self.state.arm_counts.items()},
            "arm_rewards": {f"{c}@{l}": rewards for (c, l), rewards in self.state.arm_rewards.items()},
            "trajectory": self.state.trajectory,
        }
        path.write_text(json.dumps(state_dict, indent=2), encoding="utf-8")
    
    @classmethod
    def load_state(cls, path: Path) -> "LinUCB":
        """Load bandit state from disk."""
        state_dict = json.loads(path.read_text(encoding="utf-8"))
        
        state = BanditState(
            context_dim=state_dict["context_dim"],
            alpha=state_dict["alpha"],
        )
        
        # Parse arm keys back to tuples
        def parse_arm(key: str) -> Tuple[str, int]:
            checkpoint, layer = key.split("@")
            return (checkpoint, int(layer))
        
        state.A = {
            parse_arm(k): np.array(v) for k, v in state_dict["A"].items()
        }
        state.b = {
            parse_arm(k): np.array(v) for k, v in state_dict["b"].items()
        }
        state.arm_counts = {
            parse_arm(k): v for k, v in state_dict["arm_counts"].items()
        }
        state.arm_rewards = {
            parse_arm(k): v for k, v in state_dict["arm_rewards"].items()
        }
        state.trajectory = state_dict["trajectory"]
        
        return cls(context_dim=state.context_dim, alpha=state.alpha, state=state)

