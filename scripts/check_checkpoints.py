#!/usr/bin/env python3
"""
Script to check which Pythia checkpoints are available on HuggingFace.

Usage:
    python scripts/check_checkpoints.py --model-size 14m
    python scripts/check_checkpoints.py --model-size 14m --check-revision step-1000
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from huggingface_hub import HfApi, RevisionNotFoundError
except ImportError:
    print("Error: huggingface_hub not installed. Install it with: pip install huggingface_hub")
    sys.exit(1)


def check_checkpoint_exists(model_id: str, revision: str) -> bool:
    """Check if a checkpoint exists on HuggingFace."""
    api = HfApi()
    try:
        api.repo_info(model_id, revision=revision)
        return True
    except RevisionNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking {model_id} @ {revision}: {e}")
        return False


def list_available_checkpoints(model_id: str, max_checkpoints: int = 200) -> list[str]:
    """List all available checkpoints (branches/tags) for a model."""
    api = HfApi()
    try:
        refs = api.list_repo_refs(model_id)
        checkpoints = []
        
        # Get branches (step-* checkpoints)
        for branch in refs.branches:
            if branch.name.startswith("step-") or branch.name == "main":
                checkpoints.append(branch.name)
        
        # Get tags if any
        for tag in refs.tags:
            if tag.name.startswith("step-") or tag.name == "main":
                checkpoints.append(tag.name)
        
        # Sort checkpoints (main first, then by step number)
        def sort_key(name: str) -> tuple:
            if name == "main":
                return (1, 0)  # main comes last
            try:
                step_num = int(name.split("-")[1])
                return (0, step_num)
            except (ValueError, IndexError):
                return (2, 0)  # unknown format
        
        checkpoints = sorted(set(checkpoints), key=sort_key)
        return checkpoints[:max_checkpoints]
    
    except Exception as e:
        print(f"Error listing checkpoints for {model_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Check Pythia checkpoints on HuggingFace")
    parser.add_argument("--model-size", type=str, default="14m", help="Model size (14m, 70m, 410m, etc.)")
    parser.add_argument("--check-revision", type=str, default=None, help="Check if specific revision exists")
    parser.add_argument("--org", type=str, default="EleutherAI", help="HuggingFace org")
    parser.add_argument("--list-all", action="store_true", help="List all available checkpoints")
    
    args = parser.parse_args()
    
    model_id = f"{args.org}/pythia-{args.model_size}"
    print(f"Model: {model_id}")
    print()
    
    if args.check_revision:
        # Check specific revision
        exists = check_checkpoint_exists(model_id, args.check_revision)
        if exists:
            print(f"✅ Revision '{args.check_revision}' exists")
        else:
            print(f"❌ Revision '{args.check_revision}' does NOT exist")
        return
    
    if args.list_all:
        # List all available checkpoints
        checkpoints = list_available_checkpoints(model_id)
        print(f"Found {len(checkpoints)} checkpoints:")
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"  {i}. {checkpoint}")
        
        # Show first and last few
        if len(checkpoints) > 10:
            print("\nFirst 5:")
            for cp in checkpoints[:5]:
                print(f"  - {cp}")
            print("\nLast 5:")
            for cp in checkpoints[-5:]:
                print(f"  - {cp}")
    else:
        # Quick check: try common checkpoints
        test_checkpoints = ["main", "step-1000", "step-2000", "step-10000", "step-50000", "step-100000"]
        print("Checking common checkpoints:")
        for checkpoint in test_checkpoints:
            exists = check_checkpoint_exists(model_id, checkpoint)
            status = "✅" if exists else "❌"
            print(f"  {status} {checkpoint}")


if __name__ == "__main__":
    main()

