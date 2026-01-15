#!/usr/bin/env python3
"""
Test different embedding extraction points to identify which matches the paper.

This script tests:
1. Current method: hidden_states[idx] (after complete transformer block)
2. Alternative: After post-attention layer norm (if accessible via hooks)
3. Compare results to identify which matches paper better
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.constants import pythia_model_id
import torch

def test_extraction_points(model_id: str, revision: str, layer: int, test_texts: list):
    """Test different extraction points for a given layer."""
    
    print(f"\n{'='*80}")
    print(f"Testing extraction points for layer {layer}")
    print(f"{'='*80}")
    
    results = {}
    
    # Method 1: Current implementation (after complete block)
    print(f"\nMethod 1: After complete transformer block (current)")
    embedder1 = HFHiddenStateEmbedder(
        model_id=model_id,
        revision=revision,
        pooling="mean",
        normalize=True,
        max_length=256,
        layer_index=layer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="auto"
    )
    
    try:
        embeddings1 = embedder1.encode(test_texts, batch_size=8)
        results['after_block'] = embeddings1
        print(f"  Shape: {embeddings1.shape}")
        print(f"  Norm (mean): {np.linalg.norm(embeddings1, axis=1).mean():.6f}")
        print(f"  Norm (std): {np.linalg.norm(embeddings1, axis=1).std():.6f}")
        del embedder1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
        results['after_block'] = None
    
    # Method 2: Try to extract after post-attention layer norm using hooks
    print(f"\nMethod 2: After post-attention layer norm (if accessible)")
    try:
        model = embedder1._model if hasattr(embedder1, '_model') and embedder1._model is not None else None
        if model is None:
            # Load model
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_id, revision=revision)
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
        
        # Try to extract via forward hook (if architecture allows)
        # This is architecture-dependent and may not work for all models
        print(f"  Note: This requires architecture-specific hooks")
        print(f"  For GPTNeoX, we need to hook into the layer structure")
        print(f"  Skipping for now - would require model-specific implementation")
        results['after_post_attn_norm'] = None
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")
        results['after_post_attn_norm'] = None
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', type=str, default='410m')
    parser.add_argument('--checkpoint', type=str, default='main')
    parser.add_argument('--layer', type=int, default=23)
    parser.add_argument('--test-texts', type=str, nargs='+', 
                       default=["This is a test sentence.", "Another example text."])
    args = parser.parse_args()
    
    model_id = pythia_model_id(size=args.model_size, org='EleutherAI')
    
    print("=" * 80)
    print("TESTING DIFFERENT EXTRACTION POINTS")
    print("=" * 80)
    print(f"Model: {model_id}@{args.checkpoint}")
    print(f"Layer: {args.layer}")
    print(f"Test texts: {len(args.test_texts)} sentences")
    
    results = test_extraction_points(
        model_id=model_id,
        revision=args.checkpoint,
        layer=args.layer,
        test_texts=args.test_texts
    )
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Extracted embeddings using:")
    for method, emb in results.items():
        if emb is not None:
            print(f"  {method}: {emb.shape}")
        else:
            print(f"  {method}: Not available")
    
    print(f"\nNote: To fully test extraction points, we need:")
    print(f"1. Paper's reported values for comparison")
    print(f"2. Architecture-specific hooks to extract from different points")
    print(f"3. Run full MTEB evaluation with different extraction methods")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
