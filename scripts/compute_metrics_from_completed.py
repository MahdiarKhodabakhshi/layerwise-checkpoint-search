#!/usr/bin/env python3
"""
Compute representation metrics from completed checkpoint-layer pairs.

This script extracts embeddings and computes metrics for checkpoint-layer pairs
that have completed MTEB evaluations, even in brute-force mode.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import json
import csv
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

from layer_time.corpus import build_representation_corpus, load_cached_corpus
from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.metrics import compute_representation_metrics
from layer_time.constants import pythia_model_id
import mteb


def get_completed_pairs(run_dir: Path) -> List[Tuple[str, int]]:
    """Get list of (checkpoint, layer) pairs that have completed tasks."""
    progress_files = list(run_dir.glob("progress_shard*.csv"))
    
    completed_pairs = set()
    for pf in progress_files:
        try:
            with open(pf, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    checkpoint = row.get('revision', '')
                    layer_str = row.get('layer', '')
                    if checkpoint and layer_str.isdigit():
                        completed_pairs.add((checkpoint, int(layer_str)))
        except Exception as e:
            print(f"Error reading {pf}: {e}")
    
    return sorted(completed_pairs)


def get_tasks_from_config() -> List[str]:
    """Get the list of MTEB tasks from config."""
    # Use layer_by_layer_32 preset
    tasks_preset = "layer_by_layer_32"
    try:
        # Get tasks using the preset
        tasks = mteb.get_tasks(tasks_preset=tasks_preset)
        task_names = [task.metadata.name for task in tasks]
        print(f"  Loaded {len(task_names)} tasks from preset '{tasks_preset}'")
        return task_names
    except Exception as e:
        print(f"  Error loading tasks from preset: {e}")
        # Fallback: use known task list
        print("  Using fallback task list...")
        return [
            "STS12", "STS13", "STS14", "STS15", "STS16", "STSBenchmark", "SICK-R",
            "TRECCOVID", "SciDocsRP", "ArguAna", "QuoraRetrieval", "MSMARCO",
            "NFCorpus", "HotpotQA", "FEVER", "DBpedia", "SCIDOCS",
            "TWEET_EVAL_A", "TWEET_EVAL_H", "TWEET_EVAL_I", "TWEET_EVAL_M",
            "TWEET_EVAL_P", "TWEET_EVAL_R", "AmazonCounterfactualClassification",
            "AmazonReviewsClassification", "ArxivClusteringS2S", "AskUbuntuDupQuestions",
            "BIOSSES", "Banking77Classification", "BiorxivClusteringS2S",
            "EmotionClassification", "MTOPDomainClassification", "MTOPIntentClassification",
            "ToxicConversationsClassification", "TweetSentimentExtractionClassification"
        ]


def compute_metrics_for_pair(
    checkpoint: str,
    layer: int,
    corpus: List[str],
    model_size: str,
    cfg: Dict,
) -> Dict[str, float]:
    """Compute metrics for a single checkpoint-layer pair."""
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
        print(f"  Extracting embeddings for {checkpoint} layer {layer}...")
        embeddings = embedder.encode(corpus, batch_size=cfg.get('batch_size', 64))
        
        # Compute metrics
        print(f"  Computing metrics for {checkpoint} layer {layer}...")
        metrics = compute_representation_metrics(embeddings)
        
        # Clean up
        del embedder
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return metrics
        
    except Exception as e:
        print(f"  ERROR computing metrics for {checkpoint} layer {layer}: {e}")
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, required=True, help='Path to run directory')
    parser.add_argument('--model-size', type=str, default='410m', help='Model size')
    parser.add_argument('--output', type=str, help='Output JSON file for metrics')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for embedding extraction')
    parser.add_argument('--max-examples', type=int, help='Max examples per task for corpus')
    parser.add_argument('--num-chunks', type=int, default=1, help='Total number of chunks for parallel runs (job arrays)')
    parser.add_argument('--chunk-index', type=int, default=0, help='Index of this chunk [0..num_chunks-1]')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Error: Run directory does not exist: {run_dir}")
        return 1
    
    print("=" * 80)
    print("COMPUTING REPRESENTATION METRICS FROM COMPLETED PAIRS")
    print("=" * 80)
    print()
    
    # Get completed pairs
    print("Step 1: Finding completed checkpoint-layer pairs...")
    completed_pairs = get_completed_pairs(run_dir)
    total_pairs = len(completed_pairs)
    print(f"Found {total_pairs} completed pairs")
    
    if total_pairs == 0:
        print("No completed pairs found. Exiting.")
        return 1
    
    # Optional: split work across chunks for parallel runs (e.g., Slurm job arrays)
    num_chunks = max(1, int(args.num_chunks))
    chunk_index = int(args.chunk_index)
    
    if num_chunks > 1:
        if not (0 <= chunk_index < num_chunks):
            print(f"ERROR: chunk_index must be in [0, {num_chunks-1}], got {chunk_index}")
            return 1
        
        # Compute chunk boundaries
        base = total_pairs // num_chunks
        rem = total_pairs % num_chunks
        
        starts = []
        start = 0
        for i in range(num_chunks):
            size = base + (1 if i < rem else 0)
            starts.append((start, start + size))
            start += size
        
        start_idx, end_idx = starts[chunk_index]
        chunk_pairs = completed_pairs[start_idx:end_idx]
        
        print(f"Using chunk {chunk_index+1}/{num_chunks}: "
              f"pairs [{start_idx}:{end_idx}] -> {len(chunk_pairs)} pairs")
    else:
        chunk_pairs = completed_pairs
    
    # Group by checkpoint (for summary printing)
    by_checkpoint = defaultdict(list)
    for checkpoint, layer in chunk_pairs:
        by_checkpoint[checkpoint].append(layer)
    
    print("\nCompleted pairs by checkpoint:")
    checkpoint_order = ["step137000", "step138000", "step139000", "step140000", 
                       "step141000", "step142000", "step143000", "main"]
    for cp in checkpoint_order:
        if cp in by_checkpoint:
            layers = sorted(by_checkpoint[cp])
            print(f"  {cp}: {len(layers)} layers")
    
    # Load or build corpus
    print("\nStep 2: Loading representation corpus...")
    corpus_cache = run_dir / "cache" / "representation_corpus.json"
    corpus = load_cached_corpus(corpus_cache)
    
    if corpus is None:
        print("Corpus not cached. Building from MTEB tasks...")
        tasks = get_tasks_from_config()
        if tasks is None:
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
    
    # Compute metrics for each pair assigned to this chunk
    print("\nStep 3: Computing metrics for completed pairs...")
    print("=" * 80)
    
    all_metrics = {}
    total_chunk = len(chunk_pairs)
    
    for idx, (checkpoint, layer) in enumerate(chunk_pairs, 1):
        print(f"\n[{idx}/{total_chunk}] Processing {checkpoint} layer {layer}...")
        
        metrics = compute_metrics_for_pair(
            checkpoint, layer, corpus, args.model_size, cfg
        )
        
        if metrics is not None:
            all_metrics[(checkpoint, layer)] = metrics
            print(f"  ✓ Computed metrics: {list(metrics.keys())}")
        else:
            print(f"  ✗ Failed to compute metrics")
    
    # Save results
    print("\n" + "=" * 80)
    print("Step 4: Saving results...")
    
    # Convert to serializable format
    output_data = {}
    for (checkpoint, layer), metrics in all_metrics.items():
        key = f"{checkpoint}_layer_{layer}"
        output_data[key] = {
            'checkpoint': checkpoint,
            'layer': layer,
            **metrics
        }
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = run_dir / "computed_metrics.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved metrics to: {output_path}")
    print(f"Total metrics computed: {len(all_metrics)}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Checkpoint':<15} {'Layer':<8} {'Prompt Entropy':<18} {'Dataset Entropy':<18} {'Curvature':<15} {'Effective Rank':<18}")
    print("-" * 80)
    
    for checkpoint in checkpoint_order:
        for layer in sorted(set(l for c, l in all_metrics.keys() if c == checkpoint)):
            if (checkpoint, layer) in all_metrics:
                m = all_metrics[(checkpoint, layer)]
                print(f"{checkpoint:<15} {layer:<8} {m.get('prompt_entropy', 0):<18.4f} "
                      f"{m.get('dataset_entropy', 0):<18.4f} {m.get('curvature', 0):<15.4f} "
                      f"{m.get('effective_rank', 0):<18.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
