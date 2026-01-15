#!/usr/bin/env python3
"""
Diagnostic script to analyze layer extraction and identify potential issues
comparing our implementation with the paper's methodology.
"""

import json
from pathlib import Path

def analyze_extraction():
    print("=" * 100)
    print("LAYER EXTRACTION ANALYSIS")
    print("=" * 100)
    print()
    
    print("OUR IMPLEMENTATION (from embedder.py):")
    print("-" * 80)
    print("1. Layer Indexing:")
    print("   - hidden_states[0] = input embeddings (NOT a transformer block)")
    print("   - hidden_states[1] = output after transformer block 0 (layer_index=0)")
    print("   - hidden_states[2] = output after transformer block 1 (layer_index=1)")
    print("   - ...")
    print("   - hidden_states[24] = output after transformer block 23 (layer_index=23)")
    print()
    print("   Code: idx = self.layer_index + 1")
    print("   So: layer_index=15 -> hidden_states[16] = after transformer block 15")
    print()
    
    print("2. Embedding Extraction:")
    print("   - Extraction point: output_hidden_states=True -> outputs.hidden_states")
    print("   - This gets the hidden state AFTER each transformer block")
    print("   - Pooling: mean pooling over non-padding tokens")
    print("   - Normalization: L2 normalization (if normalize=True)")
    print("   - Max length: 256 tokens")
    print()
    
    print("3. Potential Issues:")
    print("   ⚠️  Layer indexing: When we use layer_index=15, we get hidden_states[16]")
    print("      This is the output AFTER transformer block 15 (0-indexed)")
    print("      The paper might mean something different by 'layer 15'")
    print()
    print("   ⚠️  Extraction point: We extract AFTER the transformer block")
    print("      The paper might extract at a different point:")
    print("      - Before layer norm vs after layer norm")
    print("      - Before residual vs after residual")
    print("      - At a specific intermediate point")
    print()
    
    # Load our results
    avg_file = Path("/scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer/exp1_main_final9layers_v2_fixed/average_main_scores.json")
    if avg_file.exists():
        with open(avg_file, 'r') as f:
            data = json.load(f)
        
        print("OUR RESULTS:")
        print("-" * 80)
        layers = sorted([int(l) for l in data['layers'].keys()])
        scores = [data['layers'][str(l)]['average_main_score'] for l in layers]
        
        print(f"Layers: {layers}")
        print(f"Scores: {[f'{s:.4f}' for s in scores]}")
        print()
        print(f"Layer 15 (our layer_index=15): {scores[0]:.4f}")
        print(f"Layer 23 (our layer_index=23): {scores[-1]:.4f}")
        print(f"Improvement: {((scores[-1]/scores[0] - 1)*100):.2f}%")
        print()
        
        print("Expected pattern from paper:")
        print("- Layers should show progressive improvement towards deeper layers")
        print("- Final layers (21-23) typically have highest scores")
        print("- Mid-layers (15-20) might plateau or show slower improvement")
        print()
        print("Our pattern:")
        if scores[-1] > scores[0] * 1.3:
            print("✓ Shows significant improvement from layer 15 to 23 (+38%)")
        else:
            print("⚠️  Improvement is less than expected")
        
        if scores[-1] < 0.4:
            print("⚠️  Final layer score (0.42) seems low compared to typical MTEB results")
            print("   Typical range for good models: 0.5-0.8+")
            print("   This suggests either:")
            print("   - Wrong extraction point")
            print("   - Different preprocessing")
            print("   - Model checkpoint mismatch")
        else:
            print("✓ Final layer score (0.42) is in reasonable range")
    
    print()
    print("RECOMMENDATIONS:")
    print("-" * 80)
    print("1. Verify layer indexing matches paper:")
    print("   - Check paper's repository or code for exact extraction")
    print("   - Confirm if paper uses 0-indexed or 1-indexed layers")
    print()
    print("2. Check extraction point:")
    print("   - Verify if paper extracts before/after layer norm")
    print("   - Check if paper includes final layer norm")
    print()
    print("3. Compare single task performance:")
    print("   - Pick one task (e.g., STS12) and compare our scores vs paper")
    print("   - If even one task is significantly different, extraction is likely wrong")
    print()
    print("4. Verify model checkpoint:")
    print("   - Ensure we're using exact same checkpoint (revision='main')")
    print("   - Check if model has been fine-tuned or modified")
    print()

if __name__ == '__main__':
    analyze_extraction()
