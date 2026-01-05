#!/usr/bin/env python3
"""
Analyze and organize representation metrics results from completed chunk files.

This script:
1. Merges all chunk JSON files into a single dataset
2. Creates organized tables and summaries
3. Exports results in CSV format for easy analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def load_all_chunks(run_dir: Path) -> Dict:
    """Load and merge all chunk JSON files."""
    chunk_files = sorted(run_dir.glob("computed_metrics_chunk_*.json"))
    
    if not chunk_files:
        print(f"ERROR: No chunk files found in {run_dir}")
        return {}
    
    print(f"Found {len(chunk_files)} chunk files")
    
    all_data = {}
    for chunk_file in chunk_files:
        print(f"  Loading {chunk_file.name}...")
        with open(chunk_file, 'r') as f:
            chunk_data = json.load(f)
            all_data.update(chunk_data)
    
    print(f"\nTotal checkpoint-layer pairs: {len(all_data)}")
    return all_data


def create_dataframe(all_data: Dict) -> pd.DataFrame:
    """Convert merged data to DataFrame."""
    rows = []
    for key, metrics in all_data.items():
        row = {
            'checkpoint': metrics.get('checkpoint', ''),
            'layer': metrics.get('layer', -1),
            **{k: v for k, v in metrics.items() if k not in ['checkpoint', 'layer']}
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by checkpoint (with 'main' last) and layer
    def sort_key(row):
        checkpoint = row['checkpoint']
        if checkpoint == 'main':
            return (1, 999999, row['layer'])
        else:
            step_num = int(checkpoint.replace('step', '')) if checkpoint.startswith('step') else 0
            return (0, step_num, row['layer'])
    
    df['_sort_key'] = df.apply(sort_key, axis=1)
    df = df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)
    
    return df


def create_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics table."""
    metric_cols = [
        'prompt_entropy', 'dataset_entropy', 'curvature', 'effective_rank',
        'logdet_covariance', 'anisotropy', 'spectral_norm', 'mean_pairwise_cosine'
    ]
    
    # Filter to columns that exist
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    summary = df[metric_cols].describe()
    
    # Add additional statistics
    summary.loc['median'] = df[metric_cols].median()
    summary.loc['std'] = df[metric_cols].std()
    summary.loc['min'] = df[metric_cols].min()
    summary.loc['max'] = df[metric_cols].max()
    
    return summary


def create_checkpoint_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary by checkpoint."""
    metric_cols = [
        'prompt_entropy', 'dataset_entropy', 'curvature', 'effective_rank',
        'logdet_covariance', 'anisotropy', 'spectral_norm', 'mean_pairwise_cosine'
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    checkpoint_summary = df.groupby('checkpoint')[metric_cols].agg([
        'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Flatten column names
    checkpoint_summary.columns = ['_'.join(col).strip() for col in checkpoint_summary.columns.values]
    
    # Add layer count
    layer_counts = df.groupby('checkpoint')['layer'].count()
    checkpoint_summary['num_layers'] = layer_counts
    
    return checkpoint_summary.reset_index()


def create_layer_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary by layer."""
    metric_cols = [
        'prompt_entropy', 'dataset_entropy', 'curvature', 'effective_rank',
        'logdet_covariance', 'anisotropy', 'spectral_norm', 'mean_pairwise_cosine'
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]
    
    layer_summary = df.groupby('layer')[metric_cols].agg([
        'mean', 'std', 'min', 'max'
    ]).round(4)
    
    # Flatten column names
    layer_summary.columns = ['_'.join(col).strip() for col in layer_summary.columns.values]
    
    # Add checkpoint count
    checkpoint_counts = df.groupby('layer')['checkpoint'].nunique()
    layer_summary['num_checkpoints'] = checkpoint_counts
    
    return layer_summary.reset_index()


def create_top_extremes(df: pd.DataFrame, metric: str, n: int = 10) -> pd.DataFrame:
    """Get top N highest and lowest values for a metric."""
    if metric not in df.columns:
        return pd.DataFrame()
    
    top_highest = df.nlargest(n, metric)[['checkpoint', 'layer', metric]]
    top_lowest = df.nsmallest(n, metric)[['checkpoint', 'layer', metric]]
    
    top_highest['rank'] = range(1, n + 1)
    top_lowest['rank'] = range(1, n + 1)
    
    return pd.DataFrame({
        'highest': top_highest.values.tolist(),
        'lowest': top_lowest.values.tolist()
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze representation metrics results')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to run directory containing chunk JSON files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis results (default: run_dir/analysis)')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ANALYZING REPRESENTATION METRICS RESULTS")
    print("=" * 80)
    print()
    
    # Load all chunks
    print("Step 1: Loading chunk files...")
    all_data = load_all_chunks(run_dir)
    
    if not all_data:
        print("ERROR: No data loaded. Exiting.")
        return 1
    
    # Create DataFrame
    print("\nStep 2: Creating DataFrame...")
    df = create_dataframe(all_data)
    print(f"  DataFrame shape: {df.shape}")
    print(f"  Checkpoints: {df['checkpoint'].nunique()}")
    print(f"  Layers: {sorted(df['layer'].unique())}")
    
    # Save full dataset
    full_csv = output_dir / "metrics_full_dataset.csv"
    df.to_csv(full_csv, index=False)
    print(f"\n  Saved full dataset to: {full_csv}")
    
    # Summary statistics
    print("\nStep 3: Computing summary statistics...")
    summary_stats = create_summary_statistics(df)
    summary_csv = output_dir / "metrics_summary_statistics.csv"
    summary_stats.to_csv(summary_csv)
    print(f"  Saved summary statistics to: {summary_csv}")
    
    # Checkpoint summary
    print("\nStep 4: Creating checkpoint summary...")
    checkpoint_summary = create_checkpoint_summary(df)
    checkpoint_csv = output_dir / "metrics_by_checkpoint.csv"
    checkpoint_summary.to_csv(checkpoint_csv, index=False)
    print(f"  Saved checkpoint summary to: {checkpoint_csv}")
    
    # Layer summary
    print("\nStep 5: Creating layer summary...")
    layer_summary = create_layer_summary(df)
    layer_csv = output_dir / "metrics_by_layer.csv"
    layer_summary.to_csv(layer_csv, index=False)
    print(f"  Saved layer summary to: {layer_csv}")
    
    # Top extremes for key metrics
    print("\nStep 6: Computing top extremes for key metrics...")
    key_metrics = ['prompt_entropy', 'dataset_entropy', 'curvature', 'effective_rank']
    key_metrics = [m for m in key_metrics if m in df.columns]
    
    for metric in key_metrics:
        extremes = create_top_extremes(df, metric, n=10)
        if not extremes.empty:
            extremes_csv = output_dir / f"metrics_top10_{metric}.csv"
            extremes.to_csv(extremes_csv, index=False)
            print(f"  Saved top 10 {metric} to: {extremes_csv}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  1. metrics_full_dataset.csv - Complete dataset with all metrics")
    print(f"  2. metrics_summary_statistics.csv - Overall statistics")
    print(f"  3. metrics_by_checkpoint.csv - Aggregated by checkpoint")
    print(f"  4. metrics_by_layer.csv - Aggregated by layer")
    print(f"  5. metrics_top10_*.csv - Top 10 highest/lowest for key metrics")
    
    # Print quick stats
    print("\n" + "=" * 80)
    print("QUICK STATISTICS")
    print("=" * 80)
    print(f"\nTotal checkpoint-layer pairs: {len(df)}")
    print(f"Unique checkpoints: {df['checkpoint'].nunique()}")
    print(f"Unique layers: {df['layer'].nunique()}")
    print(f"Layers covered: {sorted(df['layer'].unique())}")
    
    if 'prompt_entropy' in df.columns:
        print(f"\nPrompt Entropy:")
        print(f"  Mean: {df['prompt_entropy'].mean():.4f}")
        print(f"  Std:  {df['prompt_entropy'].std():.4f}")
        print(f"  Range: [{df['prompt_entropy'].min():.4f}, {df['prompt_entropy'].max():.4f}]")
    
    if 'effective_rank' in df.columns:
        print(f"\nEffective Rank:")
        print(f"  Mean: {df['effective_rank'].mean():.4f}")
        print(f"  Std:  {df['effective_rank'].std():.4f}")
        print(f"  Range: [{df['effective_rank'].min():.4f}, {df['effective_rank'].max():.4f}]")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
