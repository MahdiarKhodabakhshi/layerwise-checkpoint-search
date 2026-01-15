#!/usr/bin/env python3
"""
Compare our results with paper's reported values.

This script helps identify:
1. Which tasks/layers differ most from paper
2. Systematic patterns in differences
3. Potential causes based on task type
"""

import sys
from pathlib import Path
import json
import pandas as pd
from collections import defaultdict

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def load_our_results(results_dir: Path, layers: list):
    """Load our experiment results."""
    results = defaultdict(lambda: defaultdict(float))
    
    for layer in layers:
        layer_dir = results_dir / f"layer_{layer:03d}"
        if not layer_dir.exists():
            continue
        
        for task_dir in layer_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            task_name = task_dir.name
            result_file = task_dir / f"{task_name}.json"
            
            if result_file.exists():
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    main_score = None
                    if 'scores' in data:
                        for subset_name, subset_data in data['scores'].items():
                            if isinstance(subset_data, list) and len(subset_data) > 0:
                                if isinstance(subset_data[0], dict):
                                    main_score = subset_data[0].get('main_score')
                                    if main_score is not None:
                                        break
                    
                    if main_score is not None:
                        results[task_name][layer] = main_score
                except Exception as e:
                    print(f"Error loading {result_file}: {e}")
    
    return results


def analyze_patterns(our_results: dict, layers: list):
    """Analyze patterns in our results."""
    print("=" * 100)
    print("OUR RESULTS ANALYSIS")
    print("=" * 100)
    
    # Group by task type
    task_types = {
        'STS': ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STS17', 'STSBenchmark', 'BIOSSES', 'SICK-R'],
        'Classification': ['AmazonCounterfactualClassification', 'AmazonReviewsClassification', 'Banking77Classification',
                          'EmotionClassification', 'MTOPDomainClassification', 'MTOPIntentClassification',
                          'MassiveIntentClassification', 'MassiveScenarioClassification', 'ToxicConversationsClassification',
                          'TweetSentimentExtractionClassification'],
        'Clustering': ['ArxivClusteringS2S', 'BiorxivClusteringS2S', 'MedrxivClusteringS2S',
                      'RedditClustering', 'StackExchangeClustering', 'TwentyNewsgroupsClustering'],
        'Reranking': ['AskUbuntuDupQuestions', 'MindSmallReranking', 'SciDocsRR', 'StackOverflowDupQuestions'],
        'Pair Classification': ['SprintDuplicateQuestions', 'TwitterSemEval2015', 'TwitterURLCorpus']
    }
    
    print("\nSummary by task type:")
    print("-" * 100)
    
    all_patterns = {}
    
    for task_type, task_list in task_types.items():
        type_tasks = [t for t in task_list if t in our_results]
        if not type_tasks:
            continue
        
        print(f"\n{task_type.upper()} ({len(type_tasks)} tasks):")
        
        # Calculate statistics per layer
        layer_stats = {}
        for layer in layers:
            scores = [our_results[t][layer] for t in type_tasks if layer in our_results[t]]
            if scores:
                layer_stats[layer] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        # Print layer-by-layer progression
        print(f"  {'Layer':<8} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print(f"  {'-'*56}")
        for layer in layers:
            if layer in layer_stats:
                stats = layer_stats[layer]
                print(f"  {layer:<8} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                      f"{stats['min']:<12.4f} {stats['max']:<12.4f}")
        
        # Calculate improvement
        if 15 in layer_stats and 23 in layer_stats:
            improvement = ((layer_stats[23]['mean'] / layer_stats[15]['mean']) - 1) * 100
            print(f"  Improvement (15→23): {improvement:+.2f}%")
        
        all_patterns[task_type] = layer_stats
    
    return all_patterns


def compare_with_paper_values(our_results: dict, paper_values: dict = None):
    """
    Compare our results with paper's reported values.
    
    paper_values format: {task_name: {layer: score}}
    If None, we'll print our values in a format ready for manual comparison.
    """
    print("\n" + "=" * 100)
    print("COMPARISON WITH PAPER (if values provided)")
    print("=" * 100)
    
    if paper_values is None:
        print("\nOur results (ready for manual comparison with paper):")
        print("\nFormat: task_name | layer_15 | layer_23")
        print("-" * 80)
        
        for task_name in sorted(our_results.keys()):
            task_scores = our_results[task_name]
            layer_15 = task_scores.get(15, None)
            layer_23 = task_scores.get(23, None)
            if layer_15 is not None or layer_23 is not None:
                layer_15_str = f"{layer_15:.6f}" if layer_15 is not None else "N/A"
                layer_23_str = f"{layer_23:.6f}" if layer_23 is not None else "N/A"
                print(f"{task_name:<45} | {layer_15_str:<12} | {layer_23_str:<12}")
        
        print("\nNote: Add paper_values dict to compare automatically")
        return
    
    # Compare if paper values provided
    print("\nComparing our results with paper values:")
    print(f"{'Task':<45} {'Layer':<8} {'Ours':<12} {'Paper':<12} {'Diff':<12} {'% Diff':<10}")
    print("-" * 100)
    
    differences = []
    
    for task_name in sorted(set(list(our_results.keys()) + list(paper_values.keys()))):
        our_task = our_results.get(task_name, {})
        paper_task = paper_values.get(task_name, {})
        
        for layer in sorted(set(list(our_task.keys()) + list(paper_task.keys()))):
            ours = our_task.get(layer, None)
            paper = paper_task.get(layer, None)
            
            if ours is not None and paper is not None:
                diff = ours - paper
                pct_diff = ((ours / paper) - 1) * 100 if paper != 0 else float('inf')
                
                print(f"{task_name:<45} {layer:<8} {ours:<12.6f} {paper:<12.6f} "
                      f"{diff:<12.6f} {pct_diff:>9.2f}%")
                
                differences.append({
                    'task': task_name,
                    'layer': layer,
                    'ours': ours,
                    'paper': paper,
                    'diff': diff,
                    'pct_diff': pct_diff
                })
    
    if differences:
        df_diff = pd.DataFrame(differences)
        print(f"\nOverall statistics:")
        print(f"  Mean absolute difference: {df_diff['diff'].abs().mean():.6f}")
        print(f"  Mean % difference: {df_diff['pct_diff'].abs().mean():.2f}%")
        print(f"  Tasks/layers where ours < paper: {(df_diff['diff'] < 0).sum()}")
        print(f"  Tasks/layers where ours > paper: {(df_diff['diff'] > 0).sum()}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory')
    parser.add_argument('--paper-values', type=str,
                       help='JSON file with paper values (optional)')
    parser.add_argument('--layers', type=int, nargs='+',
                       default=[15, 16, 17, 18, 19, 20, 21, 22, 23])
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir) / "outputs" / "mteb" / "Pythia" / "410m" / "main"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    # Load our results
    print("Loading our results...")
    our_results = load_our_results(results_dir, args.layers)
    print(f"Loaded {len(our_results)} tasks")
    
    # Analyze patterns
    patterns = analyze_patterns(our_results, args.layers)
    
    # Load paper values if provided
    paper_values = None
    if args.paper_values:
        paper_file = Path(args.paper_values)
        if paper_file.exists():
            with open(paper_file, 'r') as f:
                paper_values = json.load(f)
            print(f"\nLoaded paper values from: {paper_file}")
    
    # Compare
    compare_with_paper_values(our_results, paper_values)
    
    # Save our results in comparison-friendly format
    output_file = Path(args.results_dir) / "our_results_for_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({k: dict(v) for k, v in our_results.items()}, f, indent=2)
    print(f"\n✓ Saved our results to: {output_file}")
    print("  Use this file to compare with paper values")
    
    return 0


if __name__ == '__main__':
    import numpy as np
    sys.exit(main())
