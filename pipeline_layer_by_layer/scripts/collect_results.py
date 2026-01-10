#!/usr/bin/env python3
"""
Collect and aggregate results from both experiments.
This script extracts all metrics and creates summary reports.
"""

import sys
from pathlib import Path
import json
import pandas as pd

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from layer_time.analysis.collect_results import collect

def main():
    output_root = Path("/scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer")
    
    print("=" * 70)
    print("Collecting Results from Layer-by-Layer Pipeline")
    print("=" * 70)
    
    # Experiment 1
    exp1_dir = output_root / "exp1_main_final9layers"
    if exp1_dir.exists():
        print(f"\nExperiment 1: Main Checkpoint, Final 9 Layers")
        print(f"Directory: {exp1_dir}")
        
        try:
            df1 = collect(exp1_dir)
            output_file1 = exp1_dir / "summary.csv"
            df1.to_csv(output_file1, index=False)
            print(f"  ✅ Collected {len(df1)} results")
            print(f"  ✅ Saved to: {output_file1}")
            
            # Summary statistics
            if 'layer' in df1.columns and 'main_score' in df1.columns:
                print(f"\n  Summary by Layer:")
                layer_summary = df1.groupby('layer')['main_score'].agg(['mean', 'std', 'count'])
                print(layer_summary)
        except Exception as e:
            print(f"  ⚠️  Error collecting Experiment 1: {e}")
    else:
        print(f"\n⚠️  Experiment 1 directory not found: {exp1_dir}")
    
    # Experiment 2
    exp2_dir = output_root / "exp2_final50checkpoints_final4layers"
    if exp2_dir.exists():
        print(f"\nExperiment 2: Final 50 Checkpoints, Final 4 Layers")
        print(f"Directory: {exp2_dir}")
        
        try:
            df2 = collect(exp2_dir)
            output_file2 = exp2_dir / "summary.csv"
            df2.to_csv(output_file2, index=False)
            print(f"  ✅ Collected {len(df2)} results")
            print(f"  ✅ Saved to: {output_file2}")
            
            # Summary statistics
            if 'revision' in df2.columns and 'main_score' in df2.columns:
                print(f"\n  Summary by Checkpoint:")
                checkpoint_summary = df2.groupby('revision')['main_score'].agg(['mean', 'std', 'count'])
                print(checkpoint_summary.head(10))
                print(f"  ... (showing first 10 of {len(checkpoint_summary)} checkpoints)")
        except Exception as e:
            print(f"  ⚠️  Error collecting Experiment 2: {e}")
    else:
        print(f"\n⚠️  Experiment 2 directory not found: {exp2_dir}")
    
    print("\n" + "=" * 70)
    print("Collection complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
