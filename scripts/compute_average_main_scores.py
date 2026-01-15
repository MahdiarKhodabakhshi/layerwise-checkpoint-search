#!/usr/bin/env python3
"""
Compute average main score for each layer.

According to the paper methodology:
AvgMainScore(l) = (1/32) * Σ MainScore_t(l)

Where MainScore_t(l) is the main_score for task t using layer l embeddings.
"""

import sys
from pathlib import Path
import json
from collections import defaultdict
from typing import Dict, List, Optional

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from layer_time.constants import LAYER_BY_LAYER_MTEB_32, flatten_task_preset


def extract_main_score(result_json_path: Path) -> Optional[float]:
    """Extract main_score from MTEB result JSON file."""
    try:
        with open(result_json_path, 'r') as f:
            data = json.load(f)
        
        # MTEB v1.14.19 format: scores -> subset -> list of dicts -> main_score
        if 'scores' in data:
            scores = data['scores']
            
            # Try different subsets (test, validation, default)
            for subset_name in ['test', 'validation', 'default']:
                if subset_name in scores:
                    subset_scores = scores[subset_name]
                    if isinstance(subset_scores, list) and len(subset_scores) > 0:
                        # Get main_score from first entry
                        if isinstance(subset_scores[0], dict):
                            main_score = subset_scores[0].get('main_score')
                            if main_score is not None:
                                return float(main_score)
            
            # If no standard subset, try to find main_score in any subset
            for subset_name, subset_data in scores.items():
                if isinstance(subset_data, list) and len(subset_data) > 0:
                    if isinstance(subset_data[0], dict):
                        main_score = subset_data[0].get('main_score')
                        if main_score is not None:
                            return float(main_score)
    
    except Exception as e:
        print(f"  Warning: Could not extract main_score from {result_json_path}: {e}")
    
    return None


def compute_average_main_scores(
    output_dir: Path,
    model_size: str = "410m",
    revision: str = "main",
    layers: List[int] = None
) -> Dict[int, Dict]:
    """Compute average main score for each layer."""
    
    # Get expected task list
    all_tasks = flatten_task_preset(LAYER_BY_LAYER_MTEB_32)
    expected_num_tasks = len(all_tasks)
    
    if layers is None:
        # Auto-detect layers from directory structure
        base_path = output_dir / "outputs" / "mteb" / "Pythia" / model_size / revision
        if base_path.exists():
            layer_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("layer_")]
            layers = sorted([int(d.name.split("_")[1]) for d in layer_dirs])
        else:
            raise ValueError(f"Output directory not found: {base_path}")
    
    results = {}
    
    print(f"Computing average main scores for layers: {layers}")
    print(f"Expected tasks per layer: {expected_num_tasks}")
    print()
    
    for layer in layers:
        layer_dir = output_dir / "outputs" / "mteb" / "Pythia" / model_size / revision / f"layer_{layer:03d}"
        
        if not layer_dir.exists():
            print(f"Warning: Layer {layer} directory not found: {layer_dir}")
            continue
        
        # Extract main scores for all tasks
        task_scores = {}
        main_scores = []
        
        for task_name in all_tasks:
            task_dir = layer_dir / task_name
            result_json = task_dir / f"{task_name}.json"
            
            if result_json.exists():
                main_score = extract_main_score(result_json)
                if main_score is not None:
                    task_scores[task_name] = main_score
                    main_scores.append(main_score)
                else:
                    print(f"  Warning: No main_score found for layer {layer}, task {task_name}")
            else:
                print(f"  Warning: Result file not found: {result_json}")
        
        if len(main_scores) == 0:
            print(f"  Error: No valid main scores found for layer {layer}")
            continue
        
        # Calculate average main score
        avg_main_score = sum(main_scores) / len(main_scores)
        
        results[layer] = {
            'average_main_score': avg_main_score,
            'num_tasks': len(main_scores),
            'expected_tasks': expected_num_tasks,
            'task_scores': task_scores,
            'all_main_scores': main_scores
        }
        
        print(f"Layer {layer:2d}: Average Main Score = {avg_main_score:.6f} "
              f"({len(main_scores)}/{expected_num_tasks} tasks)")
    
    return results


def print_summary_table(results: Dict[int, Dict]):
    """Print a formatted summary table."""
    print()
    print("=" * 80)
    print("AVERAGE MAIN SCORE BY LAYER")
    print("=" * 80)
    print()
    print(f"{'Layer':<8} {'Average Main Score':<20} {'Tasks':<10} {'Status':<10}")
    print("-" * 80)
    
    for layer in sorted(results.keys()):
        data = results[layer]
        avg_score = data['average_main_score']
        num_tasks = data['num_tasks']
        expected = data['expected_tasks']
        status = "✓ Complete" if num_tasks == expected else f"⚠ Partial ({num_tasks}/{expected})"
        
        print(f"{layer:<8} {avg_score:<20.6f} {num_tasks:<10} {status:<10}")
    
    print("-" * 80)
    print()
    
    # Additional statistics
    if results:
        all_avg_scores = [data['average_main_score'] for data in results.values()]
        print(f"Mean across layers: {sum(all_avg_scores) / len(all_avg_scores):.6f}")
        print(f"Min: {min(all_avg_scores):.6f} (layer {min(results.keys(), key=lambda l: results[l]['average_main_score'])})")
        print(f"Max: {max(all_avg_scores):.6f} (layer {max(results.keys(), key=lambda l: results[l]['average_main_score'])})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute average main scores per layer")
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to experiment run directory')
    parser.add_argument('--model-size', type=str, default='410m',
                       help='Model size (default: 410m)')
    parser.add_argument('--revision', type=str, default='main',
                       help='Model revision/checkpoint (default: main)')
    parser.add_argument('--layers', type=int, nargs='+',
                       help='Specific layers to compute (default: auto-detect)')
    parser.add_argument('--output', type=str,
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1
    
    # Compute average main scores
    results = compute_average_main_scores(
        output_dir=run_dir,
        model_size=args.model_size,
        revision=args.revision,
        layers=args.layers
    )
    
    if not results:
        print("Error: No results computed")
        return 1
    
    # Print summary table
    print_summary_table(results)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            'model_size': args.model_size,
            'revision': args.revision,
            'layers': {}
        }
        
        for layer, data in sorted(results.items()):
            output_data['layers'][layer] = {
                'average_main_score': data['average_main_score'],
                'num_tasks': data['num_tasks'],
                'expected_tasks': data['expected_tasks'],
                'task_scores': data['task_scores']
            }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
