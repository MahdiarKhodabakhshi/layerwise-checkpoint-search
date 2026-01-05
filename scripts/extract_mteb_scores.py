#!/usr/bin/env python3
"""
Extract and aggregate all MTEB scores from completed runs.

This script finds all completed task evaluations and extracts:
- Main score from MTEB result files
- Task metadata
- Per-task, per-layer, per-checkpoint scores
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import csv
from collections import defaultdict
from typing import Dict, List, Optional, Any
import pandas as pd


def _try_load_json(path: Path) -> Optional[Any]:
    """Try to load JSON from a file path, return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_score_from_result_file(obj: Any) -> Optional[float]:
    """
    Extract main_score from MTEB result file structure.
    
    MTEB result files have structure:
    {
      'scores': {
        'test': [{'main_score': float, ...}, ...],
        'validation': [{'main_score': float, ...}, ...],
        ...
      },
      ...
    }
    
    Prefer test split, fallback to validation, then any split.
    """
    if obj is None or not isinstance(obj, dict):
        return None
    
    # Look for 'scores' key
    if 'scores' not in obj or not isinstance(obj['scores'], dict):
        return None
    
    scores = obj['scores']
    
    # Prefer 'test' split, then 'validation', then any split
    for split_name in ['test', 'validation', 'dev']:
        if split_name in scores and isinstance(scores[split_name], list):
            split_scores = scores[split_name]
            if len(split_scores) > 0 and isinstance(split_scores[0], dict):
                if 'main_score' in split_scores[0]:
                    score = split_scores[0]['main_score']
                    if isinstance(score, (int, float)):
                        return float(score)
    
    # Fallback: try any split
    for split_scores in scores.values():
        if isinstance(split_scores, list) and len(split_scores) > 0:
            if isinstance(split_scores[0], dict) and 'main_score' in split_scores[0]:
                score = split_scores[0]['main_score']
                if isinstance(score, (int, float)):
                    return float(score)
    
    return None


def _extract_all_scores_from_result_file(obj: Any) -> Dict[str, Any]:
    """
    Extract all available scores from MTEB result file.
    Returns a dict with split names as keys and score dicts as values.
    """
    result = {}
    if obj is None or not isinstance(obj, dict):
        return result
    
    if 'scores' not in obj or not isinstance(obj['scores'], dict):
        return result
    
    scores = obj['scores']
    for split_name, split_data in scores.items():
        if isinstance(split_data, list) and len(split_data) > 0:
            if isinstance(split_data[0], dict):
                result[split_name] = split_data[0].copy()
    
    return result


def find_result_files(run_dir: Path) -> List[Path]:
    """
    Find all MTEB result JSON files in the run directory.
    
    Looks for files matching pattern:
    outputs/mteb/Pythia/{model_size}/{checkpoint}/layer_{N}/{task}/results/EleutherAI__pythia-{size}/{checkpoint}/{task}.json
    """
    outputs_dir = run_dir / "outputs" / "mteb"
    result_files = []
    
    # Find all result JSON files in results subdirectories
    # Pattern: .../results/EleutherAI__pythia-{size}/{checkpoint}/{task}.json
    for result_file in outputs_dir.rglob("results/**/*.json"):
        # Filter for actual result files (not metadata)
        if "model_meta" not in result_file.name:
            # The structure is: .../{task}/results/EleutherAI__pythia-{size}/{checkpoint}/{task}.json
            # So the file name should match the task name (from grandparent directory)
            try:
                # Get task name from path (4 levels up: task/.../results/.../checkpoint/task.json)
                task_dir = result_file.parent.parent.parent.parent  # Go up to task directory
                task_name = task_dir.name
                if result_file.stem == task_name:
                    result_files.append(result_file)
            except (IndexError, AttributeError):
                # If we can't determine, include it anyway (will be filtered later)
                result_files.append(result_file)
    
    return sorted(result_files)


def extract_scores_from_run(run_dir: Path) -> pd.DataFrame:
    """
    Extract all MTEB scores from a run directory.
    
    Returns a DataFrame with columns:
    - model_family, model_size, revision, layer, task
    - main_score (test split)
    - all_scores (dict with all splits)
    - output_dir, result_file_path
    """
    rows = []
    
    # Find all result files
    result_files = find_result_files(run_dir)
    print(f"Found {len(result_files)} result files")
    
    # Also check done.json markers as fallback
    outputs_dir = run_dir / "outputs" / "mteb"
    done_files = list(outputs_dir.rglob("done.json"))
    print(f"Found {len(done_files)} done.json markers")
    
    processed_pairs = set()
    
    # Process result files
    for result_file in result_files:
        try:
            # Parse path: .../Pythia/410m/step137000/layer_000/STS12/results/EleutherAI__pythia-410m/step137000/STS12.json
            parts = result_file.parts
            try:
                idx = parts.index("mteb")
                model_family = parts[idx + 1]
                model_size = parts[idx + 2]
                revision = parts[idx + 3]
                layer_str = parts[idx + 4]  # layer_XXX
                task = parts[idx + 5]
                layer = int(layer_str.split("_")[1])
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse path {result_file}: {e}")
                continue
            
            pair_key = (model_family, model_size, revision, layer, task)
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            # Load result data
            result_data = _try_load_json(result_file)
            if result_data is None:
                continue
            
            # Extract main score
            main_score = _extract_score_from_result_file(result_data)
            
            # Extract all scores
            all_scores = _extract_all_scores_from_result_file(result_data)
            
            # Get output directory
            output_dir = result_file.parent.parent.parent  # Go up from results/EleutherAI__pythia-410m/step137000
            
            rows.append({
                "model_family": model_family,
                "model_size": model_size,
                "revision": revision,
                "layer": layer,
                "task": task,
                "main_score": main_score,
                "all_scores": json.dumps(all_scores) if all_scores else None,
                "task_name": result_data.get("task_name", task),
                "mteb_version": result_data.get("mteb_version", ""),
                "evaluation_time": result_data.get("evaluation_time", ""),
                "output_dir": str(output_dir),
                "result_file": str(result_file),
            })
            
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            continue
    
    # Also process done.json files that don't have result files found above
    for done_path in done_files:
        try:
            parts = done_path.parts
            try:
                idx = parts.index("mteb")
                model_family = parts[idx + 1]
                model_size = parts[idx + 2]
                revision = parts[idx + 3]
                layer_str = parts[idx + 4]  # layer_XXX
                task = parts[idx + 5]
                layer = int(layer_str.split("_")[1])
            except (ValueError, IndexError):
                continue
            
            pair_key = (model_family, model_size, revision, layer, task)
            if pair_key in processed_pairs:
                continue
            
            # Try to find result file in this directory
            out_dir = done_path.parent
            # Try multiple possible locations
            possible_result_files = [
                out_dir / f"{task}.json",
                out_dir / "results" / "EleutherAI__pythia-410m" / revision / f"{task}.json",
                out_dir / "no_model_name_available" / "no_revision_available" / f"{task}.json",
            ]
            
            result_data = None
            result_file = None
            for rf in possible_result_files:
                if rf.exists():
                    result_data = _try_load_json(rf)
                    if result_data:
                        result_file = rf
                        break
            
            if result_data is None:
                # Skip if no result file found
                continue
            
            processed_pairs.add(pair_key)
            
            main_score = _extract_score_from_result_file(result_data)
            all_scores = _extract_all_scores_from_result_file(result_data)
            done_data = _try_load_json(done_path) or {}
            
            rows.append({
                "model_family": model_family,
                "model_size": model_size,
                "revision": revision,
                "layer": layer,
                "task": task,
                "main_score": main_score,
                "all_scores": json.dumps(all_scores) if all_scores else None,
                "task_name": result_data.get("task_name", task),
                "mteb_version": result_data.get("mteb_version", ""),
                "evaluation_time": result_data.get("evaluation_time", ""),
                "output_dir": str(out_dir),
                "result_file": str(result_file) if result_file else None,
            })
            
        except Exception as e:
            print(f"Error processing {done_path}: {e}")
            continue
    
    return pd.DataFrame(rows)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract and aggregate MTEB scores from completed runs"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g., /scratch/mahdiar/pythia-layer-time-runs/20260112_181751)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file path (default: {run_dir}/mteb_scores.csv)"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional: also output as JSON file"
    )
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1
    
    print("=" * 80)
    print("EXTRACTING MTEB SCORES FROM COMPLETED RUNS")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print()
    
    # Extract scores
    df = extract_scores_from_run(run_dir)
    
    if len(df) == 0:
        print("No scores found. Exiting.")
        return 1
    
    print(f"\nExtracted {len(df)} score records")
    print(f"\nBreakdown:")
    print(f"  Unique checkpoints: {df['revision'].nunique()}")
    print(f"  Unique layers: {df['layer'].nunique()}")
    print(f"  Unique tasks: {df['task'].nunique()}")
    print(f"  Records with scores: {df['main_score'].notna().sum()}")
    
    # Determine output path
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = run_dir / "mteb_scores.csv"
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Saved scores to: {output_csv}")
    
    # Save JSON if requested
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict format
        json_data = df.to_dict(orient='records')
        with open(output_json, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        print(f"✅ Saved scores to: {output_json}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if df['main_score'].notna().any():
        print(f"\nMain Score Statistics:")
        print(f"  Mean: {df['main_score'].mean():.4f}")
        print(f"  Std:  {df['main_score'].std():.4f}")
        print(f"  Min:  {df['main_score'].min():.4f}")
        print(f"  Max:  {df['main_score'].max():.4f}")
        
        print(f"\nScores by Checkpoint:")
        checkpoint_stats = df.groupby('revision')['main_score'].agg(['mean', 'std', 'count'])
        print(checkpoint_stats.to_string())
        
        print(f"\nScores by Layer (mean across all checkpoints and tasks):")
        layer_stats = df.groupby('layer')['main_score'].agg(['mean', 'std', 'count'])
        print(layer_stats.to_string())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
