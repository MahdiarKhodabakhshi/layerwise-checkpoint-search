#!/usr/bin/env python3
"""
Script to list all available Pythia checkpoints on HuggingFace.

Usage:
    python scripts/list_available_checkpoints.py --model-size 14m
    python scripts/list_available_checkpoints.py --model-size 14m --output checkpoints_14m.txt
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from huggingface_hub import HfApi, RevisionNotFoundError
except ImportError:
    print("Error: huggingface_hub not installed. Install it with: pip install huggingface_hub")
    sys.exit(1)


def list_available_checkpoints(model_id: str) -> List[str]:
    """List all available checkpoints (branches/tags) for a model."""
    api = HfApi()
    try:
        refs = api.list_repo_refs(model_id)
        checkpoints = []
        
        # Get branches (step* checkpoints and main)
        # Pythia checkpoints are named step{NUMBER} (no hyphen), e.g., step1000, step2000
        for branch in refs.branches:
            if (branch.name.startswith("step") and branch.name[4:].isdigit()) or branch.name == "main":
                checkpoints.append(branch.name)
        
        # Get tags if any
        for tag in refs.tags:
            if (tag.name.startswith("step") and tag.name[4:].isdigit()) or tag.name == "main":
                checkpoints.append(tag.name)
        
        # Sort checkpoints (main last, step* by number)
        def sort_key(name: str) -> tuple:
            if name == "main":
                return (1, 0)  # main comes last
            if name.startswith("step"):
                try:
                    # Handle both step{NUMBER} and step-{NUMBER} formats
                    if "-" in name:
                        step_num = int(name.split("-")[1])
                    else:
                        step_num = int(name[4:])  # step{NUMBER} format
                    return (0, step_num)
                except (ValueError, IndexError):
                    return (2, 0)  # unknown format
            return (2, 0)  # unknown format
        
        checkpoints = sorted(set(checkpoints), key=sort_key)
        return checkpoints
    
    except Exception as e:
        print(f"Error listing checkpoints for {model_id}: {e}", file=sys.stderr)
        return []


def verify_checkpoint(model_id: str, revision: str) -> bool:
    """Verify if a specific checkpoint exists."""
    api = HfApi()
    try:
        api.repo_info(model_id, revision=revision)
        return True
    except RevisionNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking {model_id} @ {revision}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="List all available Pythia checkpoints on HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all checkpoints for 14m model
  python scripts/list_available_checkpoints.py --model-size 14m

  # List for all model sizes
  python scripts/list_available_checkpoints.py --model-size all

  # Save to file
  python scripts/list_available_checkpoints.py --model-size 14m --output checkpoints_14m.txt

  # Verify specific checkpoint
  python scripts/list_available_checkpoints.py --model-size 14m --verify step-1000
        """
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="14m",
        help="Model size (14m, 70m, 410m, etc.) or 'all' for all sizes"
    )
    parser.add_argument(
        "--org",
        type=str,
        default="EleutherAI",
        help="HuggingFace org (default: EleutherAI)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save checkpoint list (JSON format)"
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        help="Verify if a specific checkpoint exists (e.g., step-1000)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["list", "yaml", "json"],
        default="list",
        help="Output format (default: list)"
    )
    
    args = parser.parse_args()
    
    # Determine model sizes to check
    if args.model_size.lower() == "all":
        model_sizes = ["14m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    else:
        model_sizes = [args.model_size]
    
    results = {}
    
    for model_size in model_sizes:
        model_id = f"{args.org}/pythia-{model_size}"
        print(f"\n{'='*60}")
        print(f"Checking: {model_id}")
        print(f"{'='*60}")
        
        if args.verify:
            # Verify specific checkpoint
            exists = verify_checkpoint(model_id, args.verify)
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"\n{status}: {args.verify}")
            results[model_size] = {args.verify: exists}
            continue
        
        # List all checkpoints
        checkpoints = list_available_checkpoints(model_id)
        
        if not checkpoints:
            print(f"❌ No checkpoints found or error occurred")
            results[model_size] = []
            continue
        
        print(f"\n✅ Found {len(checkpoints)} checkpoints:")
        print()
        
        # Show first 10, last 10, and summary
        if len(checkpoints) <= 20:
            for i, cp in enumerate(checkpoints, 1):
                print(f"  {i:3d}. {cp}")
        else:
            print("First 10 checkpoints:")
            for i, cp in enumerate(checkpoints[:10], 1):
                print(f"  {i:3d}. {cp}")
            
            print(f"\n... ({len(checkpoints) - 20} more) ...\n")
            
            print("Last 10 checkpoints:")
            for i, cp in enumerate(checkpoints[-10:], len(checkpoints) - 9):
                print(f"  {i:3d}. {cp}")
        
        # Summary statistics
        step_checkpoints = [cp for cp in checkpoints if cp.startswith("step")]
        print(f"\nSummary:")
        print(f"  Total checkpoints: {len(checkpoints)}")
        print(f"  Step checkpoints: {len(step_checkpoints)}")
        print(f"  Main branch: {'Yes' if 'main' in checkpoints else 'No'}")
        
        if step_checkpoints:
            try:
                step_nums = [int(cp.split("-")[1]) for cp in step_checkpoints]
                print(f"  Min step: {min(step_nums)}")
                print(f"  Max step: {max(step_nums)}")
                print(f"  Step range: {min(step_nums)} - {max(step_nums)}")
                
                # Check for gaps
                expected_range = range(min(step_nums), max(step_nums) + 1)
                missing = [s for s in expected_range if f"step{s}" not in checkpoints]
                if missing and len(missing) <= 20:
                    print(f"  Missing steps (first 20): {missing[:20]}")
                elif missing:
                    print(f"  Missing steps: {len(missing)} total (first 20: {missing[:20]})")
            except (ValueError, IndexError):
                pass
        
        results[model_size] = checkpoints
    
    # Output in requested format
    if args.output:
        output_path = Path(args.output)
        if args.format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {output_path} (JSON format)")
        elif args.format == "yaml":
            import yaml
            with open(output_path, "w") as f:
                yaml.dump(results, f, default_flow_style=False, sort_keys=False)
            print(f"\n✅ Results saved to {output_path} (YAML format)")
        else:
            # List format - save as JSON for easy parsing
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n✅ Results saved to {output_path} (JSON format)")
    
    # Also print in requested format if output specified
    if args.format == "yaml" and not args.output:
        import yaml
        print("\n" + "="*60)
        print("YAML Format:")
        print("="*60)
        print(yaml.dump(results, default_flow_style=False, sort_keys=False))
    elif args.format == "json" and not args.output:
        print("\n" + "="*60)
        print("JSON Format:")
        print("="*60)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

