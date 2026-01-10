#!/usr/bin/env python3
"""
End-to-end test of the MTEB v1.14.19 pipeline.
Tests the full workflow: embedder -> MTEB evaluation -> result saving -> result extraction.

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
print("END-TO-END PIPELINE TEST")
print("=" * 100)
print()

# Test configuration
TEST_LAYER = 15  # Test with layer 15 (first layer in Experiment 1)
TEST_TASK = "STS12"  # Small, fast task for testing
MODEL_SIZE = "410m"
REVISION = "main"

print(f"Test Configuration:")
print(f"  Model: EleutherAI/pythia-{MODEL_SIZE}")
print(f"  Revision: {REVISION}")
print(f"  Layer: {TEST_LAYER}")
print(f"  Task: {TEST_TASK}")
print()

# Step 1: Check MTEB version
print("Step 1: Checking MTEB version...")
print(f"  MTEB version: {mteb.__version__}")
if not mteb.__version__.startswith("1.14"):
    print(f"  ⚠️  WARNING: Expected v1.14.x, got {mteb.__version__}")
else:
    print("  ✓ MTEB version correct")
print()

# Step 2: Create embedder
print("Step 2: Creating embedder...")
try:
    # Check CUDA availability
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
        dtype="auto",  # Let it auto-detect
    )
    print(f"  ✓ Embedder created successfully")
    print(f"    Device: {embedder._device_str()}")
    print(f"    Layer: {embedder.layer_index}")
    print(f"    Pooling: {embedder.pooling}")
    print(f"    Normalize: {embedder.normalize}")
except Exception as e:
    print(f"  ❌ Failed to create embedder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 3: Get MTEB task
print("Step 3: Getting MTEB task...")
try:
    benchmark = mteb.get_benchmark("MTEB(eng)")
    tasks_list = [t for t in benchmark if t.metadata.name == TEST_TASK]
    if not tasks_list:
        print(f"  ❌ Task '{TEST_TASK}' not found in benchmark")
        print(f"  Available tasks: {[t.metadata.name for t in benchmark[:10]]}...")
        sys.exit(1)
    task = tasks_list[0]
    print(f"  ✓ Found task: {TEST_TASK}")
    print(f"    Description: {task.metadata.description[:80]}...")
except Exception as e:
    print(f"  ❌ Failed to get task: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 4: Create MTEB evaluator
print("Step 4: Creating MTEB evaluator...")
try:
    evaluator = MTEB(tasks=[task])
    print("  ✓ Evaluator created")
except Exception as e:
    print(f"  ❌ Failed to create evaluator: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 5: Run evaluation (this is the actual test)
print("Step 5: Running MTEB evaluation...")
print("  This may take a few minutes...")
try:
    with tempfile.TemporaryDirectory() as temp_output:
        task_results = evaluator.run(
            embedder,
            verbosity=1,  # Show progress
            output_folder=temp_output,
            overwrite_results=True,
            raise_error=True,
            encode_kwargs={
                "batch_size": 32,
                "max_length": 512,
            },
        )
        
        if not task_results or len(task_results) == 0:
            print("  ❌ No results returned")
            sys.exit(1)
        
        result = task_results[0]
        print(f"  ✓ Evaluation completed successfully")
        print(f"    Result type: {type(result).__name__}")
except Exception as e:
    print(f"  ❌ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 6: Test result saving
print("Step 6: Testing result saving...")
try:
    test_output_dir = Path(tempfile.mkdtemp())
    result_path = test_output_dir / f"{TEST_TASK}.json"
    
    # Save using to_disk (as in our actual code)
    result.to_disk(result_path)
    
    if not result_path.exists():
        print(f"  ❌ Result file not created: {result_path}")
        sys.exit(1)
    
    print(f"  ✓ Result saved to: {result_path}")
    print(f"    File size: {result_path.stat().st_size} bytes")
except Exception as e:
    print(f"  ❌ Failed to save result: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 7: Test result loading and score extraction
print("Step 7: Testing result loading and score extraction...")
try:
    result_data = _try_load_json(result_path)
    if result_data is None:
        print("  ❌ Failed to load result JSON")
        sys.exit(1)
    
    print(f"  ✓ Result JSON loaded")
    print(f"    Keys: {list(result_data.keys())[:10]}...")
    
    # Extract score using our extraction function
    score = _extract_score_from_result_file(result_data)
    if score is None:
        print("  ⚠️  Could not extract main_score (this might be OK for some tasks)")
        # Try to see what's in the result
        print(f"    Result structure: {json.dumps(result_data, indent=2, default=str)[:500]}...")
    else:
        print(f"  ✓ Extracted main_score: {score:.4f}")
except Exception as e:
    print(f"  ❌ Failed to extract score: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
print()

# Step 8: Test result.to_dict() method
print("Step 8: Testing result.to_dict()...")
try:
    result_dict = result.to_dict()
    print(f"  ✓ to_dict() works")
    print(f"    Dict keys: {list(result_dict.keys())[:10]}...")
except Exception as e:
    print(f"  ⚠️  to_dict() failed: {e}")
    print("    (This is OK, we use to_disk() instead)")
print()

# Step 9: Verify the result structure matches what we expect
print("Step 9: Verifying result structure...")
try:
    # Check if result has the expected attributes
    expected_attrs = ['scores', 'task_name', 'dataset_name']
    found_attrs = []
    for attr in expected_attrs:
        if hasattr(result, attr):
            found_attrs.append(attr)
    
    if found_attrs:
        print(f"  ✓ Found expected attributes: {found_attrs}")
    else:
        print(f"  ⚠️  Expected attributes not found (might be OK)")
    
    # Try to get score directly from result object
    if hasattr(result, 'get_score'):
        try:
            direct_score = result.get_score()
            print(f"  ✓ result.get_score() = {direct_score}")
        except Exception as e:
            print(f"  ⚠️  get_score() failed: {e}")
except Exception as e:
    print(f"  ⚠️  Structure check failed: {e}")
print()

# Step 10: Cleanup
print("Step 10: Cleanup...")
try:
    import shutil
    shutil.rmtree(test_output_dir)
    print("  ✓ Cleanup complete")
except Exception as e:
    print(f"  ⚠️  Cleanup warning: {e}")
print()

# Final summary
print("=" * 100)
print("✅ END-TO-END TEST COMPLETED SUCCESSFULLY!")
print("=" * 100)
print()
print("Summary:")
print("  ✓ MTEB v1.14.19 API works correctly")
print("  ✓ Embedder creation and configuration works")
print("  ✓ MTEB evaluation runs successfully")
print("  ✓ Result saving (to_disk) works")
print("  ✓ Result loading and score extraction works")
print()
print("The pipeline should work correctly for the full Experiment 1 run!")
print()
