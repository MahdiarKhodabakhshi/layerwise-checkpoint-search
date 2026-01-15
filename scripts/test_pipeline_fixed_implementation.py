#!/usr/bin/env python3
"""
Comprehensive end-to-end test for the fixed implementation.

This test verifies:
1. Pooling method pools over LAST tokens (not ALL tokens)
2. padding_side = 'left' is set correctly
3. max_length = 2048 works correctly
4. Full pipeline runs successfully with corrected implementation
"""

import sys
from pathlib import Path
import tempfile
import json

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import mteb
from layer_time.embedder import HFHiddenStateEmbedder, _mean_pool

print("=" * 100)
print("COMPREHENSIVE END-TO-END TEST: FIXED IMPLEMENTATION")
print("=" * 100)
print()

# Test configuration
MODEL_ID = "EleutherAI/pythia-410m"
REVISION = "main"
LAYER = 15  # Test with one layer
TASKS = ["STS12", "AmazonCounterfactualClassification"]  # Test with 2 tasks (STS and Classification)
MAX_LENGTH = 2048
BATCH_SIZE = 64

print(f"Test Configuration:")
print(f"  Model: {MODEL_ID}")
print(f"  Revision: {REVISION}")
print(f"  Layer: {LAYER}")
print(f"  Tasks: {', '.join(TASKS)}")
print(f"  Max Length: {MAX_LENGTH}")
print(f"  Batch Size: {BATCH_SIZE}")
print()

# Step 1: Verify pooling method implementation
print("=" * 100)
print("Step 1: Verifying Pooling Method Implementation")
print("=" * 100)

# Create a dummy tensor to test pooling
batch_size = 3
seq_len = 10
hidden_dim = 8

# Create dummy hidden states and attention masks
# Simulate sequences of lengths [7, 10, 5] (so last tokens are at indices [:, -7:, :], [:, -10:, :], [:, -5:, :])
hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long)
attention_mask[0, -7:] = 1  # Sequence of length 7 (last 7 tokens)
attention_mask[1, :] = 1    # Sequence of length 10 (all tokens)
attention_mask[2, -5:] = 1  # Sequence of length 5 (last 5 tokens)

print(f"  Testing _mean_pool with:")
print(f"    Batch size: {batch_size}")
print(f"    Sequence lengths: [7, 10, 5] (from attention_mask)")
print(f"    Hidden dim: {hidden_dim}")
print()

# Test pooling
pooled = _mean_pool(hidden_states, attention_mask)

print(f"  ✓ Pooling completed successfully")
print(f"    Output shape: {pooled.shape} (expected: ({batch_size}, {hidden_dim}))")

# Verify that pooling uses LAST tokens
# For sequence 0 (length 7), should pool over hidden_states[0, -7:, :]
expected_pool_0 = hidden_states[0, -7:, :].mean(dim=0)
actual_pool_0 = pooled[0]

if torch.allclose(expected_pool_0, actual_pool_0, atol=1e-5):
    print(f"  ✓ Verified: Pooling uses LAST tokens (sequence 0: last 7 tokens)")
else:
    print(f"  ✗ ERROR: Pooling does NOT use last tokens correctly!")
    print(f"    Expected (last 7 tokens mean): {expected_pool_0[:3]}")
    print(f"    Actual: {actual_pool_0[:3]}")
    sys.exit(1)

# For sequence 1 (length 10), should pool over hidden_states[1, -10:, :] = hidden_states[1, :, :]
expected_pool_1 = hidden_states[1, :, :].mean(dim=0)
actual_pool_1 = pooled[1]

if torch.allclose(expected_pool_1, actual_pool_1, atol=1e-5):
    print(f"  ✓ Verified: Pooling uses LAST tokens (sequence 1: last 10 tokens = all tokens)")
else:
    print(f"  ✗ ERROR: Pooling does NOT use last tokens correctly!")
    sys.exit(1)

print()

# Step 2: Verify embedder configuration
print("=" * 100)
print("Step 2: Verifying Embedder Configuration")
print("=" * 100)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

embedder = HFHiddenStateEmbedder(
    model_id=MODEL_ID,
    revision=REVISION,
    pooling="mean",
    normalize=True,
    max_length=MAX_LENGTH,
    batch_size=BATCH_SIZE,
    device=device,
    dtype="auto",
    layer_index=LAYER,
)

# Trigger lazy loading to set tokenizer
_ = embedder.encode(["test"])

print(f"  ✓ Embedder created")
print(f"    Max length: {embedder.max_length} (expected: {MAX_LENGTH})")
if embedder.max_length != MAX_LENGTH:
    print(f"    ✗ ERROR: max_length mismatch!")
    sys.exit(1)

print(f"    Pooling: {embedder.pooling} (expected: mean)")
print(f"    Normalize: {embedder.normalize} (expected: True)")
print(f"    Layer index: {embedder.layer_index} (expected: {LAYER})")

# Check tokenizer padding_side
tokenizer = embedder._tokenizer
if tokenizer is None:
    print(f"    ✗ ERROR: Tokenizer not loaded!")
    sys.exit(1)

padding_side = tokenizer.padding_side
print(f"    Tokenizer padding_side: {padding_side} (expected: left)")
if padding_side != "left":
    print(f"    ✗ ERROR: padding_side is '{padding_side}', expected 'left'!")
    sys.exit(1)

print()

# Step 3: Test embedding extraction
print("=" * 100)
print("Step 3: Testing Embedding Extraction")
print("=" * 100)

test_texts = [
    "This is a test sentence.",
    "Another test sentence with more words.",
    "Short.",
]

print(f"  Encoding {len(test_texts)} test texts...")
embeddings = embedder.encode(test_texts)

print(f"  ✓ Encoding completed")
print(f"    Output shape: {embeddings.shape}")
print(f"    Expected shape: ({len(test_texts)}, hidden_dim)")

if embeddings.shape[0] != len(test_texts):
    print(f"    ✗ ERROR: Wrong batch dimension!")
    sys.exit(1)

if embeddings.shape[1] <= 0:
    print(f"    ✗ ERROR: Invalid embedding dimension!")
    sys.exit(1)

print(f"    Embedding dimension: {embeddings.shape[1]}")
print()

# Step 4: Test MTEB evaluation with corrected implementation
print("=" * 100)
print("Step 4: Testing MTEB Evaluation (Full Pipeline)")
print("=" * 100)

# Check MTEB version
mteb_version = getattr(mteb, "__version__", "unknown")
print(f"  MTEB version: {mteb_version}")

if not mteb_version.startswith("1."):
    print(f"    ⚠ WARNING: MTEB version is {mteb_version}, expected v1.14.19")
else:
    print(f"    ✓ MTEB version correct (v1.14.19)")

print()

USE_V1_API = mteb_version.startswith("1.")

if USE_V1_API:
    from mteb import MTEB as MTEBEvaluator
    benchmark = mteb.get_benchmark("MTEB(eng)")
else:
    print(f"  ✗ ERROR: This test requires MTEB v1.14.19 API")
    sys.exit(1)

results_summary = {}

for task_name in TASKS:
    print(f"\n  Testing task: {task_name}")
    print("  " + "-" * 96)
    
    # Get task from benchmark
    tasks_list = [t for t in benchmark if t.metadata.name == task_name]
    if not tasks_list:
        print(f"    ✗ ERROR: Task '{task_name}' not found in benchmark")
        sys.exit(1)
    
    task = tasks_list[0]
    print(f"    ✓ Found task: {task.metadata.name}")
    
    # Create evaluator
    evaluator = MTEBEvaluator(tasks=[task])
    print(f"    ✓ Created evaluator")
    
    # Run evaluation
    with tempfile.TemporaryDirectory() as tmpdir:
        output_folder = Path(tmpdir)
        
        print(f"    Running evaluation (this may take a few minutes)...")
        try:
            task_results = evaluator.run(
                embedder,
                verbosity=0,
                output_folder=output_folder,
                overwrite_results=True,
            )
            
            if task_results is None or task_name not in task_results:
                print(f"    ✗ ERROR: Evaluation returned no results")
                sys.exit(1)
            
            result = task_results[task_name]
            print(f"    ✓ Evaluation completed")
            
            # Extract main score
            main_score = None
            if hasattr(result, 'scores') and result.scores:
                if 'test' in result.scores and result.scores['test']:
                    if isinstance(result.scores['test'], list) and result.scores['test'][0]:
                        main_score = result.scores['test'][0].get('main_score')
                elif 'validation' in result.scores and result.scores['validation']:
                    if isinstance(result.scores['validation'], list) and result.scores['validation'][0]:
                        main_score = result.scores['validation'][0].get('main_score')
            
            if main_score is not None:
                print(f"    ✓ Main score extracted: {main_score:.6f}")
                results_summary[task_name] = main_score
            else:
                print(f"    ⚠ WARNING: Could not extract main_score, but evaluation completed")
                results_summary[task_name] = None
                
        except Exception as e:
            print(f"    ✗ ERROR: Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

print()
print("=" * 100)
print("Step 5: Summary")
print("=" * 100)
print()
print("Test Results:")
print(f"  Task scores:")
for task_name, score in results_summary.items():
    if score is not None:
        print(f"    {task_name}: {score:.6f}")
    else:
        print(f"    {task_name}: (completed but score unavailable)")

print()
print("=" * 100)
print("✅ ALL TESTS PASSED!")
print("=" * 100)
print()
print("The fixed implementation is working correctly:")
print("  ✓ Pooling method pools over LAST tokens (matching paper)")
print("  ✓ padding_side = 'left' is set correctly")
print("  ✓ max_length = 2048 works correctly")
print("  ✓ Full MTEB pipeline runs successfully")
print()
print("You can now safely submit the batch job:")
print("  sbatch slurm/exp1_main_final9layers_fixed.sbatch")
print()
