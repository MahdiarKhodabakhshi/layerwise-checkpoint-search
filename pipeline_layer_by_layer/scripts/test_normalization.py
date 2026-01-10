#!/usr/bin/env python3
"""
Quick test script to check if normalization is causing the difference.

This script re-evaluates ONE task for ONE layer with and without normalization
to see if that's the cause of the discrepancy with the paper.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.constants import pythia_model_id
import mteb
import torch

def test_normalization_effect():
    """Test one task with and without normalization."""
    
    model_size = "410m"
    checkpoint = "main"
    layer = 16
    task_name = "AmazonCounterfactualClassification"
    
    model_id = pythia_model_id(size=model_size, org="EleutherAI")
    
    print("=" * 100)
    print("TESTING NORMALIZATION EFFECT")
    print("=" * 100)
    print(f"\nModel: {model_id}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Layer: {layer}")
    print(f"Task: {task_name}")
    
    results = {}
    
    for normalize in [True, False]:
        print(f"\n{'='*100}")
        print(f"Testing with normalize={normalize}")
        print(f"{'='*100}")
        
        try:
            # Create embedder
            embedder = HFHiddenStateEmbedder(
                model_id=model_id,
                revision=checkpoint,
                pooling="mean",
                normalize=normalize,
                max_length=256,
                batch_size=64,
                device="cuda",
                dtype="auto",
                layer_index=layer,
            )
            
            # Get task
            tasks_obj = mteb.get_tasks(tasks=[task_name])
            
            # Evaluate
            print(f"Running MTEB evaluation...")
            result = mteb.evaluate(
                model=embedder,
                tasks=tasks_obj,
                show_progress_bar=True,
            )
            
            # Extract score
            if hasattr(result, 'model_dump'):
                result_dict = result.model_dump()
            else:
                result_dict = result
            
            # Extract main_score
            main_score = None
            if 'task_results' in result_dict and len(result_dict['task_results']) > 0:
                tr = result_dict['task_results'][0]
                scores = tr.get('scores', {})
                for split in ['test', 'validation']:
                    if split in scores and len(scores[split]) > 0:
                        entry = scores[split][0]
                        if 'main_score' in entry:
                            main_score = entry['main_score']
                            break
            
            results[normalize] = main_score
            print(f"✅ Main score: {main_score:.4f}")
            
            # Clean up
            del embedder
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results[normalize] = None
    
    # Compare
    print(f"\n{'='*100}")
    print("COMPARISON")
    print(f"{'='*100}")
    print(f"\nWith normalization (True):  {results.get(True, 'N/A')}")
    print(f"Without normalization (False): {results.get(False, 'N/A')}")
    
    if results.get(True) and results.get(False):
        diff = results[False] - results[True]
        pct_diff = (diff / results[True]) * 100 if results[True] > 0 else 0
        print(f"\nDifference: {diff:.4f} ({pct_diff:+.2f}%)")
        
        if abs(pct_diff) > 1:
            print(f"\n⚠️  Significant difference! Normalization affects results.")
            print(f"   This could be the cause of discrepancy with paper.")
        else:
            print(f"\n✅ Small difference. Normalization is probably NOT the cause.")

if __name__ == "__main__":
    test_normalization_effect()
