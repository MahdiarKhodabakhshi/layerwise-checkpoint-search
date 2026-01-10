#!/usr/bin/env python3
"""Generate checkpoint lists for Pythia models.

Pythia models have 154 intermediate checkpoints available.
This script generates different sampling strategies.
"""

import argparse
import sys


def generate_sparse_sampling() -> list[str]:
    """Sparse sampling: Good for initial exploration (~8 checkpoints)."""
    return [
        "step-1000",   # Very early
        "step-2000",   # Early
        "step-5000",   # Early-mid
        "step-10000",  # Mid
        "step-20000",  # Mid-late
        "step-40000",  # Late
        "step-80000",  # Very late
        "main"         # Final
    ]


def generate_uniform_sampling(interval: int = 5000) -> list[str]:
    """Uniform sampling: Every N steps.
    
    Args:
        interval: Step interval (e.g., 5000 = every 5000 steps)
    
    Returns:
        List of checkpoint names
    """
    # Pythia models typically train for ~143k steps (143,000)
    # But checkpoints might go up to different values
    # Common max: 143000, but some models might have more
    
    max_steps = 143000  # Typical Pythia training length
    checkpoints = []
    
    step = interval
    while step <= max_steps:
        checkpoints.append(f"step-{step}")
        step += interval
    
    checkpoints.append("main")  # Always include final
    return checkpoints


def generate_dense_sampling(interval: int = 1000) -> list[str]:
    """Dense sampling: Every N steps (more thorough).
    
    Args:
        interval: Step interval (e.g., 1000 = every 1000 steps)
    
    Returns:
        List of checkpoint names
    """
    return generate_uniform_sampling(interval)


def generate_log_sampling() -> list[str]:
    """Logarithmic sampling: More checkpoints early, fewer late.
    
    Good for capturing rapid early changes.
    """
    checkpoints = []
    
    # Early: every 1000 steps up to 10000
    for step in range(1000, 10001, 1000):
        checkpoints.append(f"step-{step}")
    
    # Mid: every 5000 steps from 10000 to 50000
    for step in range(15000, 50001, 5000):
        checkpoints.append(f"step-{step}")
    
    # Late: every 10000 steps from 50000 to 143000
    for step in range(60000, 143001, 10000):
        checkpoints.append(f"step-{step}")
    
    checkpoints.append("main")
    return checkpoints


def generate_all_checkpoints() -> list[str]:
    """Generate all 154 checkpoints (if we know the pattern).
    
    Note: This might not be exact - depends on actual Pythia checkpoint schedule.
    """
    # Pythia typically saves checkpoints at specific intervals
    # Common pattern: every 1000 steps, but might vary
    # This is a guess - should verify with actual model
    
    checkpoints = []
    # Assuming every 1000 steps from 1000 to 143000
    for step in range(1000, 143001, 1000):
        checkpoints.append(f"step-{step}")
    
    checkpoints.append("main")
    return checkpoints


def main():
    parser = argparse.ArgumentParser(
        description="Generate checkpoint lists for Pythia models"
    )
    parser.add_argument(
        "--strategy",
        choices=["sparse", "uniform", "dense", "log", "all"],
        default="sparse",
        help="Sampling strategy (default: sparse)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5000,
        help="Step interval for uniform/dense sampling (default: 5000)"
    )
    parser.add_argument(
        "--format",
        choices=["list", "yaml", "python"],
        default="list",
        help="Output format (default: list)"
    )
    
    args = parser.parse_args()
    
    # Generate checkpoints
    if args.strategy == "sparse":
        checkpoints = generate_sparse_sampling()
    elif args.strategy == "uniform":
        checkpoints = generate_uniform_sampling(args.interval)
    elif args.strategy == "dense":
        checkpoints = generate_dense_sampling(args.interval)
    elif args.strategy == "log":
        checkpoints = generate_log_sampling()
    elif args.strategy == "all":
        checkpoints = generate_all_checkpoints()
    
    # Output
    if args.format == "list":
        print("\n".join(checkpoints))
    elif args.format == "yaml":
        print("revisions:")
        for cp in checkpoints:
            print(f'  - "{cp}"')
    elif args.format == "python":
        print(f"revisions = {checkpoints}")
    
    # Also print summary
    print(f"\n# Generated {len(checkpoints)} checkpoints using '{args.strategy}' strategy", file=sys.stderr)


if __name__ == "__main__":
    main()

