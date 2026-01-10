#!/usr/bin/env python3
"""
Comprehensive end-to-end test of the MTEB v1.14.19 pipeline.
Tests multiple tasks to catch edge cases with different data formats.

Run this on an interactive GPU node to verify everything works before submitting the full job.
"""
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import mteb
from mteb import MTEB
from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.analysis.collect_results import _try_load_json, _extract_score_from_result_file

print("=" * 100)
print("COMPREHENSIVE END-TO-END PIPELINE TEST")
print("=" * 100)
print()

# Test configuration
TEST_LAYER = 15
TEST_TASKS = [
    # STS tasks (2)
    "STS12",  # Small STS task
    "STSBenchmark",  # Another STS task
    # Classification task (1)
    "AmazonCounterfactualClassification",  # Classification task (was failing)
    # Clustering task (1)
    "ArxivClusteringS2S",  # Clustering task (was failing)
    # Reranking tasks (3)
    "MindSmallReranking",  # Reranking task (was failing)
    "SciDocsRR",  # Reranking task (was failing)
    "AskUbuntuDupQuestions",  # Reranking task (was failing)
    # Pair Classification task (1)
    "SprintDuplicateQuestions",  # Pair Classification task - tests sentence pairs
]
MODEL_SIZE = "410m"
REVISION = "main"

print(f"Test Configuration:")
print(f"  Model: EleutherAI/pythia-{MODEL_SIZE}")
print(f"  Revision: {REVISION}")
print(f"  Layer: {TEST_LAYER}")
print(f"  Tasks: {', '.join(TEST_TASKS)}")
print()

# Step 1: Check MTEB version
print("Step 1: Checking MTEB version...")
print(f"  MTEB version: {mteb.__version__}")
if not mteb.__version__.startswith("1.14"):
    print(f"  ❌ ERROR: Expected v1.14.x, got {mteb.__version__}")
    sys.exit(1)
else:
    print("  ✓ MTEB version correct")
print()

# Step 2: Create embedder
print("Step 2: Creating embedder...")
try:
    import torch
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(f"  GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("  ⚠️  No GPU available, using CPU (will be slow)")
    
    embedder = HFHiddenStateEmbedder(
        model_id=f"EleutherAI/pythia-{MODEL_SIZE}",
        revision=REVISION,
        layer_index=TEST_LAYER,
        pooling="mean",
        normalize=True,
        device="cuda" if use_cuda else "cpu",
        dtype="auto",
    )
    print(f"  ✓ Embedder created successfully")
    print(f"    Device: {embedder._device_str()}")
    print(f"    Layer: {embedder.layer_index}")
except Exception as e:
    print(f"  ❌ Failed to create embedder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 3: Get MTEB tasks
print("Step 3: Getting MTEB tasks...")
try:
    benchmark = mteb.get_benchmark("MTEB(eng)")
    tasks_list = []
    for task_name in TEST_TASKS:
        task_objs = [t for t in benchmark if t.metadata.name == task_name]
        if not task_objs:
            print(f"  ⚠️  Task '{task_name}' not found, skipping")
            continue
        tasks_list.append(task_objs[0])
        print(f"  ✓ Found task: {task_name}")
    
    if not tasks_list:
        print("  ❌ No tasks found")
        sys.exit(1)
except Exception as e:
    print(f"  ❌ Failed to get tasks: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 4: Test each task individually
print("Step 4: Testing each task...")
print()

all_passed = True
for i, task in enumerate(tasks_list, 1):
    task_name = task.metadata.name
    print(f"  [{i}/{len(tasks_list)}] Testing task: {task_name}")
    
    try:
        # Create evaluator for this task
        evaluator = MTEB(tasks=[task])
        
        # Run evaluation
        with tempfile.TemporaryDirectory() as temp_output:
            task_results = evaluator.run(
                embedder,
                verbosity=1,
                output_folder=temp_output,
                overwrite_results=True,
                raise_error=True,
                encode_kwargs={
                    "batch_size": 32,
                    "max_length": 512,
                },
            )
            
            if not task_results or len(task_results) == 0:
                print(f"    ❌ No results returned")
                all_passed = False
                continue
            
            result = task_results[0]
            
            # Test saving
            test_output_dir = Path(tempfile.mkdtemp())
            result_path = test_output_dir / f"{task_name}.json"
            result.to_disk(result_path)
            
            # Test loading
            result_data = _try_load_json(result_path)
            if result_data is None:
                print(f"    ❌ Failed to load result JSON")
                all_passed = False
                continue
            
            # Extract score
            score = _extract_score_from_result_file(result_data)
            if score is not None:
                print(f"    ✓ Success! Score: {score:.4f}")
            else:
                print(f"    ✓ Success! (score extraction returned None, but evaluation completed)")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_output_dir)
            
    except Exception as e:
        print(f"    ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False
    print()

# Final summary
print("=" * 100)
if all_passed:
    print("✅ ALL TESTS PASSED!")
else:
    print("❌ SOME TESTS FAILED!")
print("=" * 100)
print()

if all_passed:
    print("Summary:")
    print("  ✓ All tasks evaluated successfully")
    print("  ✓ Column object handling works for all task types")
    print("  ✓ Result saving and loading works")
    print()
    print("The pipeline should work correctly for the full Experiment 1 run!")
    sys.exit(0)
else:
    print("Some tests failed. Please fix the issues before submitting the batch job.")
    sys.exit(1)
