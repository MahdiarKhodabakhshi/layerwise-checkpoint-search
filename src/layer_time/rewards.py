"""Reward computation and z-scoring.

Step 6 of the workflow: Define the reward so comparisons are consistent.

Rewards are z-scored relative to a final-checkpoint layer sweep baseline,
then aggregated across tasks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from layer_time.analysis.collect_results import collect


def compute_baseline_scores(
    run_dir: Path,
    final_checkpoint: str,
    model_size: str,
    layers: List[int],
    tasks: List[str],
) -> Dict[str, Dict[int, float]]:
    """
    Compute baseline scores from final checkpoint layer sweep.
    
    Args:
        run_dir: Directory containing MTEB results
        final_checkpoint: Checkpoint revision (e.g., "main")
        model_size: Model size (e.g., "410m")
        layers: List of layer indices
        tasks: List of task names
    
    Returns:
        Dictionary: task_name -> {layer: score}
    """
    # Load results using collect_results
    df = collect(run_dir)
    
    # Filter for final checkpoint
    baseline_df = df[
        (df["revision"] == final_checkpoint)
        & (df["model_size"] == model_size)
        & (df["layer"].isin(layers))
        & (df["task"].isin(tasks))
        & (df["score"].notna())
    ].copy()
    
    baseline_scores: Dict[str, Dict[int, float]] = {}
    
    for task in tasks:
        task_df = baseline_df[baseline_df["task"] == task]
        if len(task_df) == 0:
            continue
        
        baseline_scores[task] = {}
        for layer in layers:
            layer_scores = task_df[task_df["layer"] == layer]["score"].values
            if len(layer_scores) > 0:
                # Use mean score across layers as baseline (or best layer)
                # Following workflow: "relative to final-checkpoint layer sweep"
                baseline_scores[task][layer] = float(np.mean(layer_scores))
        
        # If no layer-specific scores, use overall task mean
        if not baseline_scores[task]:
            all_scores = task_df["score"].values
            if len(all_scores) > 0:
                mean_score = float(np.mean(all_scores))
                baseline_scores[task] = {layer: mean_score for layer in layers}
    
    return baseline_scores


def compute_z_scored_reward(
    raw_score: float,
    baseline_scores: Dict[int, float],
    method: str = "mean",
) -> float:
    """
    Compute z-scored reward relative to baseline.
    
    Args:
        raw_score: Raw MTEB score for the (checkpoint, layer, task)
        baseline_scores: Dict mapping layer -> baseline score for final checkpoint
        method: How to aggregate baseline ("mean", "max", "min")
    
    Returns:
        Z-scored reward (float)
    """
    if not baseline_scores:
        return 0.0
    
    baseline_values = list(baseline_scores.values())
    
    if method == "mean":
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
    elif method == "max":
        baseline_mean = np.max(baseline_values)
        baseline_std = np.std(baseline_values) if len(baseline_values) > 1 else 1.0
    elif method == "min":
        baseline_mean = np.min(baseline_values)
        baseline_std = np.std(baseline_values) if len(baseline_values) > 1 else 1.0
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if baseline_std < 1e-10:
        # If no variance, return 0 (or simple difference)
        return float(raw_score - baseline_mean)
    
    # Z-score: (x - mean) / std
    z_score = (raw_score - baseline_mean) / baseline_std
    return float(z_score)


def aggregate_rewards(
    rewards: Dict[str, float],
    method: str = "mean",
) -> float:
    """
    Aggregate rewards across tasks.
    
    Args:
        rewards: Dictionary mapping task_name -> reward
        method: Aggregation method ("mean", "harmonic_mean", "robust_mean")
    
    Returns:
        Aggregated reward (float)
    """
    if not rewards:
        return 0.0
    
    reward_values = list(rewards.values())
    
    if method == "mean":
        return float(np.mean(reward_values))
    elif method == "harmonic_mean":
        # Robust to outliers
        positive_rewards = [r for r in reward_values if r > 0]
        if positive_rewards:
            return float(len(positive_rewards) / np.sum(1.0 / np.array(positive_rewards)))
        return float(np.mean(reward_values))
    elif method == "robust_mean":
        # Use median as robust mean
        return float(np.median(reward_values))
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_task_reward(
    raw_score: float,
    task_name: str,
    baseline_scores: Dict[str, Dict[int, float]],
    layer: int,
) -> float:
    """
    Compute z-scored reward for a single task.
    
    Args:
        raw_score: Raw MTEB score
        task_name: Name of the task
        baseline_scores: Baseline scores from final checkpoint
        layer: Layer index
    
    Returns:
        Z-scored reward
    """
    if task_name not in baseline_scores:
        return 0.0
    
    task_baseline = baseline_scores[task_name]
    
    # Use layer-specific baseline if available, otherwise use mean
    if layer in task_baseline:
        baseline_score = task_baseline[layer]
        baseline_std = np.std(list(task_baseline.values())) if len(task_baseline) > 1 else 1.0
    else:
        baseline_score = np.mean(list(task_baseline.values()))
        baseline_std = np.std(list(task_baseline.values())) if len(task_baseline) > 1 else 1.0
    
    if baseline_std < 1e-10:
        return float(raw_score - baseline_score)
    
    return float((raw_score - baseline_score) / baseline_std)