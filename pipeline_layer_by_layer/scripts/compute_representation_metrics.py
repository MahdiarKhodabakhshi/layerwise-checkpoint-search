#!/usr/bin/env python3
"""
Compute representation metrics for all checkpoint-layer pairs.
This matches the methodology from the information_flow repository:
https://github.com/OFSkean/information_flow

Metrics computed:
- Prompt entropy
- Dataset entropy
- Curvature
- Effective rank
"""

import sys
from pathlib import Path
import json

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from layer_time.corpus import build_representation_corpus, load_cached_corpus
from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.metrics import compute_representation_metrics
from layer_time.constants import pythia_model_id
import yaml

def main():
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pipeline_layer_by_layer/configs/exp1_main_final9layers.yaml")
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    run_dir = Path(config['run']['output_root']) / config['run']['run_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Computing Representation Metrics")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Output: {run_dir}")
    print("")
    
    # Build or load corpus
    corpus_cache_path = run_dir / "cache" / "representation_corpus.json"
    if corpus_cache_path.exists():
        print("Loading cached corpus...")
        corpus = load_cached_corpus(corpus_cache_path)
    else:
        print("Building representation corpus...")
        corpus = build_representation_corpus(
            model_sizes=config['hf']['model_sizes'],
            tasks_preset=config['mteb']['tasks_preset'],
            max_examples_per_task=config.get('metrics', {}).get('corpus_max_examples_per_task'),
        )
        corpus_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(corpus_cache_path, 'w') as f:
            json.dump(corpus, f)
        print(f"Corpus saved to: {corpus_cache_path}")
    
    print(f"Corpus size: {len(corpus)} examples")
    print("")
    
    # Get layers
    model_size = config['hf']['model_sizes'][0]
    model_id = pythia_model_id(model_size, config['hf']['org'])
    
    # Create embedder to get layer count
    embedder = HFHiddenStateEmbedder(
        model_id=model_id,
        revision=config['hf']['revisions'][0],
        layer_index=0,  # Will be set per layer
        pooling=config['embedding']['pooling'],
        normalize=config['embedding']['normalize'],
        max_length=config['embedding']['max_length'],
        batch_size=config['embedding']['batch_size'],
        device=config['embedding']['device'],
        dtype=config['embedding']['dtype'],
    )
    
    # Parse layers
    layers_spec = config['embedding']['layers']
    if layers_spec == "all":
        layers = list(range(embedder.num_hidden_layers))
    else:
        layers = [int(l.strip()) for l in layers_spec.split(",") if l.strip()]
    
    print(f"Computing metrics for:")
    print(f"  Model: {model_id}")
    print(f"  Checkpoints: {config['hf']['revisions']}")
    print(f"  Layers: {layers}")
    print("")
    
    # Compute metrics for each checkpoint-layer pair
    metrics_dir = run_dir / "cache" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for revision in config['hf']['revisions']:
        for layer in layers:
            cache_file = metrics_dir / f"{revision}_layer_{layer:03d}.json"
            
            if cache_file.exists():
                print(f"  ✓ {revision} layer {layer}: Using cached metrics")
                with open(cache_file) as f:
                    metrics = json.load(f)
            else:
                print(f"  Computing {revision} layer {layer}...")
                
                # Create embedder for this checkpoint-layer
                embedder = HFHiddenStateEmbedder(
                    model_id=model_id,
                    revision=revision,
                    layer_index=layer,
                    pooling=config['embedding']['pooling'],
                    normalize=config['embedding']['normalize'],
                    max_length=config['embedding']['max_length'],
                    batch_size=config['embedding']['batch_size'],
                    device=config['embedding']['device'],
                    dtype=config['embedding']['dtype'],
                )
                
                # Extract embeddings
                embeddings = embedder.encode(corpus)
                
                # Compute metrics
                metrics = compute_representation_metrics(embeddings)
                
                # Save
                with open(cache_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            results.append({
                'model_size': model_size,
                'revision': revision,
                'layer': layer,
                **metrics
            })
    
    # Save summary
    import pandas as pd
    df = pd.DataFrame(results)
    summary_file = run_dir / "representation_metrics.csv"
    df.to_csv(summary_file, index=False)
    
    print("")
    print("=" * 70)
    print("Metrics computation complete!")
    print("=" * 70)
    print(f"Results saved to: {summary_file}")
    print(f"Total pairs: {len(results)}")
    print("")
    print("Metrics computed:")
    print("  - prompt_entropy")
    print("  - dataset_entropy")
    print("  - curvature")
    print("  - effective_rank")

if __name__ == "__main__":
    main()
