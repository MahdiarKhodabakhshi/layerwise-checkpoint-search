#!/usr/bin/env python3
"""
Compute representation metrics for Experiment 1 layers.

This script computes representation metrics for layers 15-23 of Pythia-410m main checkpoint.
"""

import sys
from pathlib import Path
import json
from typing import Dict, List

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from layer_time.corpus import build_representation_corpus, load_cached_corpus
from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.metrics import compute_representation_metrics
from layer_time.constants import pythia_model_id
import mteb


def get_tasks_from_preset() -> List[str]:
    """Get the list of MTEB tasks from preset."""
    try:
        tasks = mteb.get_tasks(tasks_preset="layer_by_layer_32")
        task_names = [task.metadata.name for task in tasks]
        return task_names
    except Exception as e:
        print(f"Error loading tasks from preset: {e}")
        # Fallback: use known task list from constants
        from layer_time.constants import LAYER_BY_LAYER_MTEB_32, flatten_task_preset
        return flatten_task_preset(LAYER_BY_LAYER_MTEB_32)


def compute_metrics_for_layer(
    layer: int,
    corpus: List[str],
    model_size: str = "410m",
    checkpoint: str = "main",
    cfg: Dict = None
) -> Dict[str, float]:
    """Compute representation metrics for a single layer."""
    if cfg is None:
        cfg = {
            'hf_org': 'EleutherAI',
            'pooling': 'mean',
            'normalize': True,
            'max_length': 256,
            'batch_size': 64,
            'device': 'cuda',
            'dtype': 'auto',
        }
    
    model_id = pythia_model_id(size=model_size, org=cfg.get('hf_org', 'EleutherAI'))
    
    try:
        # Create embedder
        embedder = HFHiddenStateEmbedder(
            model_id=model_id,
            revision=checkpoint,
            pooling=cfg.get('pooling', 'mean'),
            normalize=cfg.get('normalize', True),
            max_length=cfg.get('max_length', 256),
            batch_size=cfg.get('batch_size', 64),
            device=cfg.get('device', 'cuda'),
            dtype=cfg.get('dtype', 'auto'),
            layer_index=layer,
        )
        
        # Extract embeddings
        print(f"  Extracting embeddings for layer {layer}...")
        embeddings = embedder.encode(corpus, batch_size=cfg.get('batch_size', 64))
        
        # Compute metrics
        print(f"  Computing metrics for layer {layer}...")
        metrics = compute_representation_metrics(embeddings)
        
        # Clean up
        del embedder
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
        
    except Exception as e:
        print(f"  ERROR computing metrics for layer {layer}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True, help='Path to run directory')
    parser.add_argument('--model-size', type=str, default='410m', help='Model size')
    parser.add_argument('--checkpoint', type=str, default='main', help='Checkpoint/revision')
    parser.add_argument('--layers', type=int, nargs='+', default=[15, 16, 17, 18, 19, 20, 21, 22, 23],
                       help='Layers to compute (default: 15-23)')
    parser.add_argument('--output', type=str, help='Output JSON file for metrics')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding extraction')
    parser.add_argument('--max-examples', type=int, default=1000, help='Max examples per task for corpus')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1
    
    print("=" * 80)
    print("COMPUTING REPRESENTATION METRICS FOR EXPERIMENT 1")
    print("=" * 80)
    print()
    
    # Load or build corpus
    print("Step 1: Loading representation corpus...")
    corpus_cache = run_dir / "cache" / "representation_corpus.json"
    corpus = load_cached_corpus(corpus_cache)
    
    if corpus is None:
        print("Corpus not cached. Building from MTEB tasks...")
        tasks = get_tasks_from_preset()
        if tasks is None or len(tasks) == 0:
            print("ERROR: Could not determine tasks. Cannot build corpus.")
            return 1
        
        print(f"Building corpus from {len(tasks)} tasks...")
        corpus = build_representation_corpus(
            tasks,
            split="train",
            max_examples_per_task=args.max_examples,
            cache_path=corpus_cache,
        )
        print(f"Built corpus: {len(corpus)} examples")
    else:
        print(f"Loaded cached corpus: {len(corpus)} examples")
    
    if len(corpus) == 0:
        print("ERROR: Corpus is empty. Cannot compute metrics.")
        return 1
    
    # Configuration
    cfg = {
        'hf_org': 'EleutherAI',
        'pooling': 'mean',
        'normalize': True,
        'max_length': 256,
        'batch_size': args.batch_size,
        'device': 'cuda',
        'dtype': 'auto',
    }
    
    # Compute metrics for each layer
    print(f"\nStep 2: Computing metrics for {len(args.layers)} layers...")
    print("=" * 80)
    
    all_metrics = {}
    
    for idx, layer in enumerate(sorted(args.layers), 1):
        print(f"\n[{idx}/{len(args.layers)}] Processing layer {layer}...")
        
        metrics = compute_metrics_for_layer(
            layer, corpus, args.model_size, args.checkpoint, cfg
        )
        
        if metrics is not None:
            all_metrics[layer] = metrics
            print(f"  ✓ Computed metrics: {list(metrics.keys())}")
        else:
            print(f"  ✗ Failed to compute metrics")
    
    # Save results
    print("\n" + "=" * 80)
    print("Step 3: Saving results...")
    
    # Convert to serializable format
    output_data = {
        'model_size': args.model_size,
        'checkpoint': args.checkpoint,
        'layers': {}
    }
    
    for layer, metrics in sorted(all_metrics.items()):
        output_data['layers'][layer] = {
            'layer': layer,
            **metrics
        }
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "computed_representation_metrics.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved metrics to: {output_path}")
    print(f"Total metrics computed: {len(all_metrics)}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("REPRESENTATION METRICS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Layer':<8} {'Prompt Entropy':<18} {'Dataset Entropy':<18} {'Curvature':<15} {'Effective Rank':<18}")
    print("-" * 80)
    
    for layer in sorted(all_metrics.keys()):
        m = all_metrics[layer]
        print(f"{layer:<8} {m.get('prompt_entropy', 0):<18.4f} "
              f"{m.get('dataset_entropy', 0):<18.4f} {m.get('curvature', 0):<15.4f} "
              f"{m.get('effective_rank', 0):<18.4f}")
    
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
