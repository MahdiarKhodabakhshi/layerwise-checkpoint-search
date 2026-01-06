#!/usr/bin/env python3
"""
Extract task-specific tables for fair comparison across layers and checkpoints.

For each task that is present across all (revision, layer) combinations,
creates a table with columns: layer, checkpoint (revision), main_score.

This allows fair analysis of whether each (layer, checkpoint) pair performs
better than others for a specific task.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Set, List, Tuple


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the expanded CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def find_common_tasks(df: pd.DataFrame, last_n_layers: int = None) -> List[str]:
    """
    Find tasks that are present for ALL (revision, layer) combinations.
    
    Args:
        df: DataFrame with columns: revision, layer, task, main_score
            (should already be filtered to desired layers)
        last_n_layers: For documentation only
    
    Returns:
        List of task names that are complete across all combinations
    """
    print("\n" + "="*80)
    print("Finding Common Tasks Across All Layers and Checkpoints")
    print("="*80)
    
    # Get all unique revisions and layers
    all_revisions = sorted(df['revision'].unique())
    all_layers = sorted(df['layer'].unique())
    
    print(f"Revisions: {all_revisions}")
    print(f"Layers: {all_layers}")
    print(f"Total combinations: {len(all_revisions)} × {len(all_layers)} = {len(all_revisions) * len(all_layers)}")
    
    # For each task, check if it has data for ALL (revision, layer) combinations
    common_tasks = []
    
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task].copy()
        
        # Get unique (revision, layer) pairs for this task
        task_combinations = set(
            zip(task_df['revision'], task_df['layer'])
        )
        
        # Expected combinations
        expected_combinations = set(
            (rev, layer) for rev in all_revisions for layer in all_layers
        )
        
        # Check if task has all combinations
        missing = expected_combinations - task_combinations
        
        if len(missing) == 0:
            common_tasks.append(task)
            print(f"  ✓ {task}: Complete ({len(task_combinations)} combinations)")
        else:
            print(f"  ✗ {task}: Missing {len(missing)} combinations (has {len(task_combinations)}, expected {len(expected_combinations)})")
            if len(missing) <= 5:
                print(f"    Missing: {sorted(missing)}")
    
    print(f"\nFound {len(common_tasks)} common tasks:")
    for task in common_tasks:
        print(f"  - {task}")
    
    return common_tasks


def create_task_table(df: pd.DataFrame, task: str, output_dir: Path, last_n_layers: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a table for a specific task with columns: layer, checkpoint, main_score.
    
    Also creates a pivoted version for easier comparison.
    
    Args:
        df: Full DataFrame (should already be filtered to desired layers)
        task: Task name
        output_dir: Directory to save tables
        last_n_layers: For documentation only
    
    Returns:
        Tuple of (long_format_df, wide_format_df)
    """
    # Filter to this task
    task_df = df[df['task'] == task].copy()
    
    # Select columns: layer, revision (checkpoint), main_score
    task_table = task_df[['layer', 'revision', 'main_score']].copy()
    task_table = task_table.sort_values(['layer', 'revision'])
    
    # Create pivoted version: layer as rows, revision as columns
    task_table_wide = task_table.pivot(
        index='layer',
        columns='revision',
        values='main_score'
    )
    
    # Sort columns by revision order (handle step numbers properly)
    def sort_key(rev):
        if rev == 'main':
            return (1, 0)  # Put 'main' at the end
        elif rev.startswith('step'):
            try:
                step_num = int(rev.replace('step', ''))
                return (0, step_num)
            except:
                return (0, 0)
        else:
            return (0, 0)
    
    # Sort columns
    sorted_cols = sorted(task_table_wide.columns, key=sort_key)
    task_table_wide = task_table_wide[sorted_cols]
    
    # Save both formats
    task_safe = task.replace('/', '_').replace(' ', '_')
    
    long_path = output_dir / f"task_{task_safe}_long.csv"
    wide_path = output_dir / f"task_{task_safe}_wide.csv"
    
    task_table.to_csv(long_path, index=False)
    task_table_wide.to_csv(wide_path)
    
    return task_table, task_table_wide


def create_summary_table(common_tasks: List[str], df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Create a summary table showing which tasks are common and their statistics.
    """
    print("\n" + "="*80)
    print("Creating Summary Table")
    print("="*80)
    
    summary_rows = []
    
    for task in common_tasks:
        task_df = df[df['task'] == task].copy()
        
        summary_rows.append({
            'task': task,
            'n_combinations': len(task_df),
            'n_revisions': task_df['revision'].nunique(),
            'n_layers': task_df['layer'].nunique(),
            'mean_score': task_df['main_score'].mean(),
            'std_score': task_df['main_score'].std(),
            'min_score': task_df['main_score'].min(),
            'max_score': task_df['main_score'].max(),
            'score_range': task_df['main_score'].max() - task_df['main_score'].min()
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('task')
    
    summary_path = output_dir / "task_tables_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(summary_df.to_string(index=False))
    
    return summary_df


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract task-specific tables for fair comparison"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV file path (expanded MTEB scores)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for tables (default: task_tables subdirectory of input)'
    )
    parser.add_argument(
        '--last-n-layers',
        type=int,
        default=None,
        help='Only consider last N layers (default: all layers)'
    )
    parser.add_argument(
        '--include-main',
        action='store_true',
        help='Include "main" checkpoint in analysis (default: exclude)'
    )
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return 1
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "task_tables"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING TASK-SPECIFIC TABLES")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    if args.last_n_layers:
        print(f"Last N layers: {args.last_n_layers}")
    if not args.include_main:
        print("Excluding 'main' checkpoint")
    print()
    
    # Load data
    df = load_data(input_path)
    
    # Exclude 'main' checkpoint if requested
    if not args.include_main:
        df = df[df['revision'] != 'main'].copy()
        print(f"Excluded 'main' checkpoint. Remaining revisions: {sorted(df['revision'].unique())}")
    
    # Filter to last N layers if specified (before finding common tasks)
    if args.last_n_layers:
        all_layers = sorted(df['layer'].unique())
        last_layers = all_layers[-args.last_n_layers:]
        df = df[df['layer'].isin(last_layers)].copy()
        print(f"\nFiltered to last {args.last_n_layers} layers: {last_layers}")
    
    # Find common tasks
    common_tasks = find_common_tasks(df, args.last_n_layers)
    
    if len(common_tasks) == 0:
        print("\n⚠️  No common tasks found across all combinations!")
        print("This might mean:")
        print("  - Tasks are missing for some (revision, layer) pairs")
        print("  - Try using --last-n-layers to restrict to a subset")
        return 1
    
    # Create tables for each common task
    print("\n" + "="*80)
    print(f"Creating Tables for {len(common_tasks)} Common Tasks")
    print("="*80)
    
    created_tables = []
    
    for task in common_tasks:
        print(f"\nProcessing: {task}")
        long_df, wide_df = create_task_table(df, task, output_dir, args.last_n_layers)
        
        task_safe = task.replace('/', '_').replace(' ', '_')
        created_tables.append({
            'task': task,
            'long_file': f"task_{task_safe}_long.csv",
            'wide_file': f"task_{task_safe}_wide.csv",
            'n_rows': len(long_df),
            'n_layers': long_df['layer'].nunique(),
            'n_checkpoints': long_df['revision'].nunique()
        })
        
        print(f"  ✓ Created long format: {len(long_df)} rows")
        print(f"  ✓ Created wide format: {wide_df.shape[0]} layers × {wide_df.shape[1]} checkpoints")
        print(f"  ✓ Score range: {long_df['main_score'].min():.4f} - {long_df['main_score'].max():.4f}")
    
    # Create summary table
    summary_df = create_summary_table(common_tasks, df, output_dir)
    
    # Print final summary
    print("\n" + "="*80)
    print("✅ ALL TASK TABLES EXTRACTED SUCCESSFULLY")
    print("="*80)
    print(f"\nTables saved to: {output_dir}")
    print(f"\nGenerated {len(common_tasks)} task tables:")
    print("\n  Long format (columns: layer, checkpoint, main_score):")
    for entry in created_tables:
        print(f"    - {entry['long_file']}")
    print("\n  Wide format (pivoted: layer × checkpoint):")
    for entry in created_tables:
        print(f"    - {entry['wide_file']}")
    print(f"\n  Summary: task_tables_summary.csv")
    
    print("\n" + "="*80)
    print("USAGE NOTES")
    print("="*80)
    print("""
Each task has two table formats:

1. Long format (task_<name>_long.csv):
   - Columns: layer, checkpoint (revision), main_score
   - Use for: Statistical analysis, plotting with grouping
   - Easy to filter, group, and aggregate

2. Wide format (task_<name>_wide.csv):
   - Rows: layers
   - Columns: checkpoints (revisions)
   - Use for: Quick visual comparison, heatmaps
   - Easy to see patterns across layers and checkpoints

Example analysis:
  - Which (layer, checkpoint) pair has the highest score?
  - Does layer 23 consistently outperform layer 16 across all checkpoints?
  - Which checkpoint shows the most improvement with deeper layers?
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
