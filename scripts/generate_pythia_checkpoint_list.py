#!/usr/bin/env python3
"""
Generate checkpoint list for Pythia models based on actual available checkpoints.

Pythia checkpoints are named step{NUMBER} (no hyphen), e.g., step1000, step2000.
This script generates a list of checkpoints based on the actual naming convention.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Based on actual git ls-remote output for pythia-70m
# Checkpoints follow a pattern: step{NUMBER} where NUMBER can be:
# - Small numbers: step0, step1, step2, step4, step8, step16, step32, step64, step128, step256, step512
# - Thousands: step1000, step2000, step3000, etc. (every 1000 from 1000 to 143000)
# - Tens of thousands: step10000, step11000, step12000, etc.
# - Hundreds of thousands: step100000, step101000, step102000, etc.
# - main (final checkpoint)

def generate_all_checkpoints():
    """Generate list of all Pythia checkpoints."""
    checkpoints = []
    
    # Small power-of-2-like checkpoints
    small_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    checkpoints.extend([f"step{s}" for s in small_steps])
    
    # Every 1000 steps from 1000 to 143000
    for step in range(1000, 144000, 1000):
        checkpoints.append(f"step{step}")
    
    # Add main
    checkpoints.append("main")
    
    return checkpoints


def generate_dense_checkpoints(interval=1000):
    """Generate dense checkpoint list (every N steps)."""
    checkpoints = []
    
    # Include small steps
    small_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    checkpoints.extend([f"step{s}" for s in small_steps if s < interval])
    
    # Every interval steps from interval to 143000
    for step in range(interval, 144000, interval):
        checkpoints.append(f"step{step}")
    
    # Add main
    checkpoints.append("main")
    
    return checkpoints


def generate_uniform_checkpoints(interval=2000):
    """Generate uniform checkpoint list (every N steps, skipping small ones)."""
    checkpoints = []
    
    # Start from interval
    for step in range(interval, 144000, interval):
        checkpoints.append(f"step{step}")
    
    # Add main
    checkpoints.append("main")
    
    return checkpoints


def generate_logarithmic_checkpoints():
    """Generate logarithmically spaced checkpoints."""
    checkpoints = []
    
    # Small steps
    small_steps = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    checkpoints.extend([f"step{s}" for s in small_steps])
    
    # Logarithmic spacing: 1k, 2k, 4k, 8k, 16k, 32k, 64k, 128k
    log_steps = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    checkpoints.extend([f"step{s}" for s in log_steps])
    
    # Add main
    checkpoints.append("main")
    
    return checkpoints


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate checkpoint list for Pythia models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all checkpoints (154 total)
  python scripts/generate_pythia_checkpoint_list.py --strategy all

  # Generate dense checkpoints (every 1000 steps)
  python scripts/generate_pythia_checkpoint_list.py --strategy dense --interval 1000

  # Generate uniform checkpoints (every 2000 steps)
  python scripts/generate_pythia_checkpoint_list.py --strategy uniform --interval 2000

  # Generate logarithmic checkpoints
  python scripts/generate_pythia_checkpoint_list.py --strategy log
        """
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["all", "dense", "uniform", "log"],
        default="dense",
        help="Checkpoint selection strategy (default: dense)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1000,
        help="Step interval for dense/uniform strategies (default: 1000)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["yaml", "json", "list"],
        default="yaml",
        help="Output format (default: yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Generate checkpoints
    if args.strategy == "all":
        checkpoints = generate_all_checkpoints()
    elif args.strategy == "dense":
        checkpoints = generate_dense_checkpoints(args.interval)
    elif args.strategy == "uniform":
        checkpoints = generate_uniform_checkpoints(args.interval)
    elif args.strategy == "log":
        checkpoints = generate_logarithmic_checkpoints()
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")
    
    # Format output
    if args.format == "yaml":
        output = "revisions: [" + ", ".join([f'"{cp}"' for cp in checkpoints]) + "]"
    elif args.format == "json":
        import json
        output = json.dumps(checkpoints, indent=2)
    else:  # list
        output = "\n".join(checkpoints)
    
    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"✅ Generated {len(checkpoints)} checkpoints and saved to {args.output}")
    else:
        print(output)
        print(f"\n# Total: {len(checkpoints)} checkpoints", file=sys.stderr)


if __name__ == "__main__":
    main()
