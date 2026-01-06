#!/usr/bin/env python3
"""
Create a human-readable summary report of the metrics analysis.
"""

import pandas as pd
from pathlib import Path
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_metrics_summary_report.py <analysis_dir>")
        return 1
    
    analysis_dir = Path(sys.argv[1])
    
    # Load key files
    df_full = pd.read_csv(analysis_dir / "metrics_full_dataset.csv")
    df_checkpoint = pd.read_csv(analysis_dir / "metrics_by_checkpoint.csv")
    df_layer = pd.read_csv(analysis_dir / "metrics_by_layer.csv")
    
    report = []
    report.append("=" * 80)
    report.append("REPRESENTATION METRICS ANALYSIS SUMMARY")
    report.append("=" * 80)
    report.append("")
    
    # Dataset overview
    report.append("DATASET OVERVIEW")
    report.append("-" * 80)
    report.append(f"Total checkpoint-layer pairs analyzed: {len(df_full)}")
    report.append(f"Unique checkpoints: {df_full['checkpoint'].nunique()}")
    report.append(f"Unique layers: {df_full['layer'].nunique()}")
    report.append(f"Layers covered: {sorted(df_full['layer'].unique())}")
    report.append("")
    
    # Key findings by metric
    report.append("KEY METRICS SUMMARY")
    report.append("-" * 80)
    
    metrics_info = {
        'prompt_entropy': 'Measures information content in embeddings',
        'dataset_entropy': 'Measures diversity of representations',
        'curvature': 'Measures geometric complexity (higher = more curved)',
        'effective_rank': 'Measures dimensionality of representation space',
        'anisotropy': 'Measures directional bias in embeddings',
        'mean_pairwise_cosine': 'Measures average similarity between embeddings'
    }
    
    for metric, description in metrics_info.items():
        if metric in df_full.columns:
            mean_val = df_full[metric].mean()
            std_val = df_full[metric].std()
            min_val = df_full[metric].min()
            max_val = df_full[metric].max()
            report.append(f"\n{metric.replace('_', ' ').title()}:")
            report.append(f"  Description: {description}")
            report.append(f"  Mean: {mean_val:.4f} ± {std_val:.4f}")
            report.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    
    report.append("")
    
    # Checkpoint trends
    report.append("CHECKPOINT TRENDS")
    report.append("-" * 80)
    report.append("\nEffective Rank by Checkpoint:")
    for _, row in df_checkpoint.iterrows():
        checkpoint = row['checkpoint']
        mean_rank = row['effective_rank_mean']
        std_rank = row['effective_rank_std']
        num_layers = int(row['num_layers'])
        report.append(f"  {checkpoint:15s}: {mean_rank:.2f} ± {std_rank:.2f} (n={num_layers} layers)")
    
    report.append("")
    
    # Layer trends
    report.append("LAYER TRENDS")
    report.append("-" * 80)
    report.append("\nEffective Rank by Layer (showing key layers):")
    key_layers = [0, 1, 2, 3, 5, 10, 15, 20, 22, 23]
    for layer in key_layers:
        if layer in df_layer['layer'].values:
            row = df_layer[df_layer['layer'] == layer].iloc[0]
            mean_rank = row['effective_rank_mean']
            std_rank = row['effective_rank_std']
            num_checkpoints = int(row['num_checkpoints'])
            report.append(f"  Layer {layer:2d}: {mean_rank:.2f} ± {std_rank:.2f} (n={num_checkpoints} checkpoints)")
    
    report.append("")
    
    # Observations
    report.append("KEY OBSERVATIONS")
    report.append("-" * 80)
    
    # Find highest/lowest effective rank
    max_rank_row = df_full.loc[df_full['effective_rank'].idxmax()]
    min_rank_row = df_full.loc[df_full['effective_rank'].idxmin()]
    
    report.append(f"\n1. Highest Effective Rank: {max_rank_row['effective_rank']:.2f}")
    report.append(f"   Checkpoint: {max_rank_row['checkpoint']}, Layer: {max_rank_row['layer']}")
    
    report.append(f"\n2. Lowest Effective Rank: {min_rank_row['effective_rank']:.2f}")
    report.append(f"   Checkpoint: {min_rank_row['checkpoint']}, Layer: {min_rank_row['layer']}")
    
    # Layer pattern
    early_layers = df_layer[df_layer['layer'].isin([0, 1, 2, 3])]['effective_rank_mean'].mean()
    late_layers = df_layer[df_layer['layer'].isin([20, 21, 22, 23])]['effective_rank_mean'].mean()
    
    report.append(f"\n3. Effective Rank Pattern:")
    report.append(f"   Early layers (0-3): {early_layers:.2f}")
    report.append(f"   Late layers (20-23): {late_layers:.2f}")
    report.append(f"   Difference: {early_layers - late_layers:.2f} (early layers have higher rank)")
    
    # Curvature extremes
    if 'curvature' in df_full.columns:
        high_curv = df_full.nlargest(1, 'curvature').iloc[0]
        low_curv = df_full.nsmallest(1, 'curvature').iloc[0]
        report.append(f"\n4. Curvature Extremes:")
        report.append(f"   Highest: {high_curv['curvature']:.2f} ({high_curv['checkpoint']}, layer {high_curv['layer']})")
        report.append(f"   Lowest: {low_curv['curvature']:.2f} ({low_curv['checkpoint']}, layer {low_curv['layer']})")
    
    report.append("")
    report.append("=" * 80)
    report.append("END OF SUMMARY")
    report.append("=" * 80)
    
    # Print and save
    report_text = "\n".join(report)
    print(report_text)
    
    report_file = analysis_dir / "metrics_summary_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print(f"\n\nFull report saved to: {report_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
