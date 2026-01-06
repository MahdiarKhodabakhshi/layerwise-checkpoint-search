#!/usr/bin/env python3
"""
Extract analysis tables from expanded MTEB scores CSV.

Generates all tables requested for analysis:
- Coverage and comparability tables
- Core performance tables
- Task-level robustness tables
- Task-specific metric tables
"""

import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the expanded CSV file."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# ============================================================================
# A. Coverage and Comparability Tables
# ============================================================================

def table_1_revision_task_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 1: Revision-task coverage table
    
    Grain: revision
    Columns: n_tasks, task_list
    """
    print("\n" + "="*80)
    print("Table 1: Revision-Task Coverage")
    print("="*80)
    
    coverage = df.groupby('revision').agg({
        'task': ['nunique', lambda x: sorted(x.unique().tolist())]
    }).reset_index()
    
    coverage.columns = ['revision', 'n_tasks', 'task_list']
    coverage['task_list'] = coverage['task_list'].apply(lambda x: ', '.join(x))
    
    print(coverage.to_string(index=False))
    return coverage


def table_2_revision_layer_coverage(df: pd.DataFrame, last_n_layers: int = 8) -> pd.DataFrame:
    """
    Table 2: Revision-layer coverage table (restricted to last N layers)
    
    Grain: revision, layer (with layer ∈ {16..23} for last 8)
    Columns: n_tasks, missing_tasks
    """
    print("\n" + "="*80)
    print(f"Table 2: Revision-Layer Coverage (last {last_n_layers} layers)")
    print("="*80)
    
    # Get last N layers
    all_layers = sorted(df['layer'].unique())
    last_layers = all_layers[-last_n_layers:]
    print(f"Using layers: {last_layers}")
    
    df_last = df[df['layer'].isin(last_layers)].copy()
    
    # For each revision, get the common task set
    revision_tasks = {}
    for rev in df_last['revision'].unique():
        tasks = set(df_last[df_last['revision'] == rev]['task'].unique())
        revision_tasks[rev] = tasks
    
    # Find common tasks across all revisions in last layers
    if revision_tasks:
        common_tasks = set.intersection(*revision_tasks.values())
        print(f"Common tasks across all revisions: {len(common_tasks)}")
    else:
        common_tasks = set()
    
    # Build coverage table
    rows = []
    for rev in sorted(df_last['revision'].unique()):
        for layer in sorted(last_layers):
            rev_layer_df = df_last[(df_last['revision'] == rev) & (df_last['layer'] == layer)]
            n_tasks = rev_layer_df['task'].nunique()
            tasks = set(rev_layer_df['task'].unique())
            missing = common_tasks - tasks
            
            rows.append({
                'revision': rev,
                'layer': layer,
                'n_tasks': n_tasks,
                'missing_tasks': ', '.join(sorted(missing)) if missing else 'none'
            })
    
    coverage = pd.DataFrame(rows)
    print(coverage.to_string(index=False))
    return coverage


def table_3_comparable_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 3: Comparable slices definition table
    
    Defines the slices for consistent analysis.
    """
    print("\n" + "="*80)
    print("Table 3: Comparable Slices Definition")
    print("="*80)
    
    # Find common tasks across all revisions
    all_revisions = sorted(df['revision'].unique())
    revision_task_sets = {}
    for rev in all_revisions:
        revision_task_sets[rev] = set(df[df['revision'] == rev]['task'].unique())
    
    # Common tasks across all revisions
    common_all = set.intersection(*revision_task_sets.values())
    
    # Common tasks between step138000 and step139000
    if 'step138000' in revision_task_sets and 'step139000' in revision_task_sets:
        common_138_139 = revision_task_sets['step138000'] & revision_task_sets['step139000']
    else:
        common_138_139 = set()
    
    # Tasks for step137000
    step137000_tasks = revision_task_sets.get('step137000', set())
    
    slices = [
        {
            'slice_id': 'S1',
            'description': 'All checkpoints, tasks = common 7',
            'revisions': ', '.join(all_revisions),
            'tasks': ', '.join(sorted(common_all)),
            'n_tasks': len(common_all),
            'n_revisions': len(all_revisions)
        },
        {
            'slice_id': 'S2',
            'description': 'step138000 vs step139000, tasks = common 12',
            'revisions': 'step138000, step139000',
            'tasks': ', '.join(sorted(common_138_139)),
            'n_tasks': len(common_138_139),
            'n_revisions': 2
        },
        {
            'slice_id': 'S3',
            'description': 'step137000 only, tasks = 32',
            'revisions': 'step137000',
            'tasks': ', '.join(sorted(step137000_tasks)),
            'n_tasks': len(step137000_tasks),
            'n_revisions': 1
        }
    ]
    
    slices_df = pd.DataFrame(slices)
    print(slices_df.to_string(index=False))
    return slices_df


# ============================================================================
# B. Core Performance Tables
# ============================================================================

def get_slice_data(df: pd.DataFrame, slice_id: str) -> pd.DataFrame:
    """Get data for a specific slice."""
    if slice_id == 'S1':
        # All revisions, common tasks
        all_revisions = sorted(df['revision'].unique())
        common_tasks = set.intersection(*[set(df[df['revision'] == r]['task'].unique()) 
                                         for r in all_revisions])
        return df[(df['revision'].isin(all_revisions)) & (df['task'].isin(common_tasks))].copy()
    elif slice_id == 'S2':
        # step138000 vs step139000, common tasks
        common_tasks = (set(df[df['revision'] == 'step138000']['task'].unique()) &
                       set(df[df['revision'] == 'step139000']['task'].unique()))
        return df[(df['revision'].isin(['step138000', 'step139000'])) & 
                  (df['task'].isin(common_tasks))].copy()
    elif slice_id == 'S3':
        # step137000 only, all tasks
        return df[df['revision'] == 'step137000'].copy()
    else:
        raise ValueError(f"Unknown slice_id: {slice_id}")


def table_4_arm_summary(df: pd.DataFrame, slice_id: str = 'S1', last_n_layers: int = 8) -> pd.DataFrame:
    """
    Table 4: Arm summary table
    
    Arm = (revision, layer), using a fixed task-set and last N layers.
    Grain: revision, layer (layers 16-23)
    """
    print("\n" + "="*80)
    print(f"Table 4: Arm Summary (Slice {slice_id}, last {last_n_layers} layers)")
    print("="*80)
    
    slice_df = get_slice_data(df, slice_id)
    
    # Get last N layers
    all_layers = sorted(slice_df['layer'].unique())
    last_layers = all_layers[-last_n_layers:]
    slice_df = slice_df[slice_df['layer'].isin(last_layers)].copy()
    
    # Aggregate by revision and layer
    arm_summary = slice_df.groupby(['revision', 'layer']).agg({
        'main_score': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).reset_index()
    
    arm_summary.columns = ['revision', 'layer', 'mean_main_score', 'median_main_score', 
                          'std_over_tasks', 'min_task_score', 'max_task_score', 'n_tasks']
    
    # Sort by revision and layer
    arm_summary = arm_summary.sort_values(['revision', 'layer'])
    
    print(arm_summary.to_string(index=False))
    return arm_summary


def table_5_layer_profile_per_checkpoint(df: pd.DataFrame, slice_id: str = 'S1', 
                                        last_n_layers: int = 8) -> pd.DataFrame:
    """
    Table 5: Layer profile per checkpoint (last-8 only)
    
    Grain: revision, layer
    Columns: mean_main_score plus rank_within_revision
    """
    print("\n" + "="*80)
    print(f"Table 5: Layer Profile Per Checkpoint (Slice {slice_id}, last {last_n_layers} layers)")
    print("="*80)
    
    slice_df = get_slice_data(df, slice_id)
    
    # Get last N layers
    all_layers = sorted(slice_df['layer'].unique())
    last_layers = all_layers[-last_n_layers:]
    slice_df = slice_df[slice_df['layer'].isin(last_layers)].copy()
    
    # Compute mean score per revision-layer
    layer_profile = slice_df.groupby(['revision', 'layer'])['main_score'].mean().reset_index()
    layer_profile.columns = ['revision', 'layer', 'mean_main_score']
    
    # Add rank within revision
    layer_profile['rank_within_revision'] = (
        layer_profile.groupby('revision')['mean_main_score']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    
    layer_profile = layer_profile.sort_values(['revision', 'layer'])
    
    print(layer_profile.to_string(index=False))
    return layer_profile


def table_6_best_layer_per_revision(df: pd.DataFrame, slice_id: str = 'S1', 
                                    last_n_layers: int = 8) -> pd.DataFrame:
    """
    Table 6: Best-layer-per-revision table
    
    Grain: revision
    Columns: best_layer, best_mean_main_score, margin_to_2nd_best
    """
    print("\n" + "="*80)
    print(f"Table 6: Best Layer Per Revision (Slice {slice_id}, last {last_n_layers} layers)")
    print("="*80)
    
    # Get layer profile
    layer_profile = table_5_layer_profile_per_checkpoint(df, slice_id, last_n_layers)
    
    # Find best layer per revision
    best_layers = []
    for rev in sorted(layer_profile['revision'].unique()):
        rev_data = layer_profile[layer_profile['revision'] == rev].sort_values('mean_main_score', ascending=False)
        
        if len(rev_data) >= 2:
            best = rev_data.iloc[0]
            second = rev_data.iloc[1]
            margin = best['mean_main_score'] - second['mean_main_score']
        elif len(rev_data) == 1:
            best = rev_data.iloc[0]
            margin = 0.0
        else:
            continue
        
        best_layers.append({
            'revision': rev,
            'best_layer': int(best['layer']),
            'best_mean_main_score': best['mean_main_score'],
            'margin_to_2nd_best': margin
        })
    
    best_layers_df = pd.DataFrame(best_layers)
    print(best_layers_df.to_string(index=False))
    return best_layers_df


def table_7_checkpoint_trend_per_layer(df: pd.DataFrame, slice_id: str = 'S1', 
                                       last_n_layers: int = 8) -> pd.DataFrame:
    """
    Table 7: Checkpoint trend table per layer
    
    Grain: layer, revision (layers 16-23)
    Columns: mean_main_score
    """
    print("\n" + "="*80)
    print(f"Table 7: Checkpoint Trend Per Layer (Slice {slice_id}, last {last_n_layers} layers)")
    print("="*80)
    
    slice_df = get_slice_data(df, slice_id)
    
    # Get last N layers
    all_layers = sorted(slice_df['layer'].unique())
    last_layers = all_layers[-last_n_layers:]
    slice_df = slice_df[slice_df['layer'].isin(last_layers)].copy()
    
    # Compute mean per layer-revision
    trend = slice_df.groupby(['layer', 'revision'])['main_score'].mean().reset_index()
    trend.columns = ['layer', 'revision', 'mean_main_score']
    
    # Pivot to wide format for easier reading
    trend_wide = trend.pivot(index='layer', columns='revision', values='mean_main_score')
    
    print(trend_wide.to_string())
    return trend, trend_wide


# ============================================================================
# C. Task-Level Robustness Tables
# ============================================================================

def table_8_task_sensitivity(df: pd.DataFrame, slice_id: str = 'S1', 
                             last_n_layers: int = 8) -> pd.DataFrame:
    """
    Table 8: Task sensitivity table across last 8 layers
    
    Grain: task (within a slice)
    Columns: range_over_layers, std_over_layers, best_layer, best_score
    """
    print("\n" + "="*80)
    print(f"Table 8: Task Sensitivity (Slice {slice_id}, last {last_n_layers} layers)")
    print("="*80)
    
    slice_df = get_slice_data(df, slice_id)
    
    # Get last N layers
    all_layers = sorted(slice_df['layer'].unique())
    last_layers = all_layers[-last_n_layers:]
    slice_df = slice_df[slice_df['layer'].isin(last_layers)].copy()
    
    # Compute statistics per task
    task_stats = []
    for task in sorted(slice_df['task'].unique()):
        task_df = slice_df[slice_df['task'] == task]
        
        # Group by layer and get mean per layer
        layer_means = task_df.groupby('layer')['main_score'].mean()
        
        if len(layer_means) > 0:
            task_stats.append({
                'task': task,
                'range_over_layers': float(layer_means.max() - layer_means.min()),
                'std_over_layers': float(layer_means.std()),
                'best_layer': int(layer_means.idxmax()),
                'best_score': float(layer_means.max()),
                'mean_score': float(layer_means.mean()),
                'best_minus_mean': float(layer_means.max() - layer_means.mean())
            })
    
    sensitivity_df = pd.DataFrame(task_stats)
    sensitivity_df = sensitivity_df.sort_values('range_over_layers', ascending=False)
    
    print(sensitivity_df.to_string(index=False))
    return sensitivity_df


def table_9_win_rate(df: pd.DataFrame, slice_id: str = 'S1', last_n_layers: int = 8,
                     baseline_revision: str = 'step137000', baseline_layer: int = 23) -> pd.DataFrame:
    """
    Table 9: Win-rate table against a baseline arm
    
    Grain: revision, layer
    Columns: wins, losses, ties, win_rate across tasks
    """
    print("\n" + "="*80)
    print(f"Table 9: Win Rate vs Baseline ({baseline_revision}, layer {baseline_layer})")
    print(f"        (Slice {slice_id}, last {last_n_layers} layers)")
    print("="*80)
    
    slice_df = get_slice_data(df, slice_id)
    
    # Get last N layers
    all_layers = sorted(slice_df['layer'].unique())
    last_layers = all_layers[-last_n_layers:]
    slice_df = slice_df[slice_df['layer'].isin(last_layers)].copy()
    
    # Get baseline scores per task
    baseline_df = slice_df[(slice_df['revision'] == baseline_revision) & 
                          (slice_df['layer'] == baseline_layer)]
    baseline_scores = dict(zip(baseline_df['task'], baseline_df['main_score']))
    
    # Compare each arm to baseline
    win_rate_rows = []
    for rev in sorted(slice_df['revision'].unique()):
        for layer in sorted(last_layers):
            arm_df = slice_df[(slice_df['revision'] == rev) & (slice_df['layer'] == layer)]
            
            wins = 0
            losses = 0
            ties = 0
            
            for task in arm_df['task'].unique():
                arm_score = arm_df[arm_df['task'] == task]['main_score'].iloc[0]
                baseline_score = baseline_scores.get(task)
                
                if baseline_score is None:
                    continue
                
                if arm_score > baseline_score:
                    wins += 1
                elif arm_score < baseline_score:
                    losses += 1
                else:
                    ties += 1
            
            total = wins + losses + ties
            win_rate = wins / total if total > 0 else 0.0
            
            win_rate_rows.append({
                'revision': rev,
                'layer': layer,
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'total_tasks': total,
                'win_rate': win_rate
            })
    
    win_rate_df = pd.DataFrame(win_rate_rows)
    win_rate_df = win_rate_df.sort_values(['revision', 'layer'])
    
    print(win_rate_df.to_string(index=False))
    return win_rate_df


# ============================================================================
# D. Task-Specific Metric Tables
# ============================================================================

def table_10_metric_availability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 10: Metric availability catalog
    
    Grain: task
    Columns: list/count of score_* columns that are non-null for that task
    """
    print("\n" + "="*80)
    print("Table 10: Metric Availability Catalog")
    print("="*80)
    
    # Get all score_* columns
    score_cols = [c for c in df.columns if c.startswith('score_')]
    
    availability_rows = []
    for task in sorted(df['task'].unique()):
        task_df = df[df['task'] == task]
        
        available_metrics = []
        for col in score_cols:
            if task_df[col].notna().any():
                available_metrics.append(col.replace('score_', ''))
        
        availability_rows.append({
            'task': task,
            'n_metrics': len(available_metrics),
            'available_metrics': ', '.join(sorted(available_metrics))
        })
    
    availability_df = pd.DataFrame(availability_rows)
    availability_df = availability_df.sort_values('n_metrics', ascending=False)
    
    print(availability_df.to_string(index=False))
    return availability_df


def table_11_long_format_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 11: Long-format metrics table parsed from all_scores
    
    Grain: revision, layer, task, metric_name
    Columns: metric_value
    """
    print("\n" + "="*80)
    print("Table 11: Long-Format Metrics Table")
    print("="*80)
    
    def extract_scores_from_json(all_scores_str: str) -> Dict[str, Any]:
        """Extract all scores from JSON string."""
        if pd.isna(all_scores_str) or not all_scores_str:
            return {}
        try:
            scores_dict = json.loads(all_scores_str)
            # Get test split
            if 'test' in scores_dict and isinstance(scores_dict['test'], dict):
                return scores_dict['test']
            # Fallback to any split
            for split_name, split_data in scores_dict.items():
                if isinstance(split_data, dict):
                    return split_data
        except:
            pass
        return {}
    
    long_format_rows = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get('all_scores')):
            scores = extract_scores_from_json(str(row['all_scores']))
            
            for metric_name, metric_value in scores.items():
                # Skip non-numeric values and lists
                if isinstance(metric_value, (int, float)):
                    long_format_rows.append({
                        'revision': row['revision'],
                        'layer': row['layer'],
                        'task': row['task'],
                        'metric_name': metric_name,
                        'metric_value': metric_value
                    })
                elif isinstance(metric_value, list) and len(metric_value) > 0:
                    # For lists, take mean if numeric
                    numeric_values = [v for v in metric_value if isinstance(v, (int, float))]
                    if numeric_values:
                        long_format_rows.append({
                            'revision': row['revision'],
                            'layer': row['layer'],
                            'task': row['task'],
                            'metric_name': metric_name,
                            'metric_value': float(sum(numeric_values) / len(numeric_values))
                        })
    
    long_format_df = pd.DataFrame(long_format_rows)
    
    print(f"Created long-format table with {len(long_format_df)} rows")
    print(f"Unique metrics: {long_format_df['metric_name'].nunique()}")
    print(f"\nSample (first 20 rows):")
    print(long_format_df.head(20).to_string(index=False))
    
    return long_format_df


# ============================================================================
# Main execution
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract analysis tables from MTEB scores")
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output-dir', type=str, help='Output directory for tables (default: same as input)')
    parser.add_argument('--slice', type=str, default='S1', choices=['S1', 'S2', 'S3'],
                       help='Slice to use for performance tables (default: S1)')
    parser.add_argument('--last-n-layers', type=int, default=8, help='Number of last layers to analyze (default: 8)')
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        return 1
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / "analysis_tables"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("EXTRACTING ANALYSIS TABLES FROM MTEB SCORES")
    print("="*80)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Slice: {args.slice}")
    print(f"Last N layers: {args.last_n_layers}")
    print()
    
    # Load data
    df = load_data(input_path)
    
    # A. Coverage and Comparability Tables
    print("\n" + "="*80)
    print("A. COVERAGE AND COMPARABILITY TABLES")
    print("="*80)
    
    table_1 = table_1_revision_task_coverage(df)
    table_1.to_csv(output_dir / "table_1_revision_task_coverage.csv", index=False)
    
    table_2 = table_2_revision_layer_coverage(df, args.last_n_layers)
    table_2.to_csv(output_dir / "table_2_revision_layer_coverage.csv", index=False)
    
    table_3 = table_3_comparable_slices(df)
    table_3.to_csv(output_dir / "table_3_comparable_slices.csv", index=False)
    
    # B. Core Performance Tables
    print("\n" + "="*80)
    print("B. CORE PERFORMANCE TABLES")
    print("="*80)
    
    table_4 = table_4_arm_summary(df, args.slice, args.last_n_layers)
    table_4.to_csv(output_dir / f"table_4_arm_summary_slice{args.slice}.csv", index=False)
    
    table_5 = table_5_layer_profile_per_checkpoint(df, args.slice, args.last_n_layers)
    table_5.to_csv(output_dir / f"table_5_layer_profile_slice{args.slice}.csv", index=False)
    
    table_6 = table_6_best_layer_per_revision(df, args.slice, args.last_n_layers)
    table_6.to_csv(output_dir / f"table_6_best_layer_per_revision_slice{args.slice}.csv", index=False)
    
    table_7_long, table_7_wide = table_7_checkpoint_trend_per_layer(df, args.slice, args.last_n_layers)
    table_7_long.to_csv(output_dir / f"table_7_checkpoint_trend_long_slice{args.slice}.csv", index=False)
    table_7_wide.to_csv(output_dir / f"table_7_checkpoint_trend_wide_slice{args.slice}.csv")
    
    # C. Task-Level Robustness Tables
    print("\n" + "="*80)
    print("C. TASK-LEVEL ROBUSTNESS TABLES")
    print("="*80)
    
    table_8 = table_8_task_sensitivity(df, args.slice, args.last_n_layers)
    table_8.to_csv(output_dir / f"table_8_task_sensitivity_slice{args.slice}.csv", index=False)
    
    table_9 = table_9_win_rate(df, args.slice, args.last_n_layers)
    table_9.to_csv(output_dir / f"table_9_win_rate_slice{args.slice}.csv", index=False)
    
    # D. Task-Specific Metric Tables
    print("\n" + "="*80)
    print("D. TASK-SPECIFIC METRIC TABLES")
    print("="*80)
    
    table_10 = table_10_metric_availability(df)
    table_10.to_csv(output_dir / "table_10_metric_availability.csv", index=False)
    
    table_11 = table_11_long_format_metrics(df)
    table_11.to_csv(output_dir / "table_11_long_format_metrics.csv", index=False)
    
    print("\n" + "="*80)
    print("✅ ALL TABLES EXTRACTED SUCCESSFULLY")
    print("="*80)
    print(f"\nTables saved to: {output_dir}")
    print("\nGenerated tables:")
    print("  A. Coverage and Comparability:")
    print("     - table_1_revision_task_coverage.csv")
    print("     - table_2_revision_layer_coverage.csv")
    print("     - table_3_comparable_slices.csv")
    print("  B. Core Performance:")
    print(f"     - table_4_arm_summary_slice{args.slice}.csv")
    print(f"     - table_5_layer_profile_slice{args.slice}.csv")
    print(f"     - table_6_best_layer_per_revision_slice{args.slice}.csv")
    print(f"     - table_7_checkpoint_trend_long_slice{args.slice}.csv")
    print(f"     - table_7_checkpoint_trend_wide_slice{args.slice}.csv")
    print("  C. Task-Level Robustness:")
    print(f"     - table_8_task_sensitivity_slice{args.slice}.csv")
    print(f"     - table_9_win_rate_slice{args.slice}.csv")
    print("  D. Task-Specific Metrics:")
    print("     - table_10_metric_availability.csv")
    print("     - table_11_long_format_metrics.csv")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
