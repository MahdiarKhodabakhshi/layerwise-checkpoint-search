#!/usr/bin/env python3
"""
Quick test script to verify MTEB v1.14.19 API works correctly.
Run this on an interactive node to test before submitting the full job.
"""
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import mteb
from mteb import MTEB
from layer_time.embedder import HFHiddenStateEmbedder

print("=" * 80)
print("MTEB v1.14.19 API Test")
print("=" * 80)
print(f"MTEB version: {mteb.__version__}")
print()

# Test 1: Check MTEB import
print("✓ MTEB imported successfully")

# Test 2: Get benchmark
benchmark = mteb.get_benchmark("MTEB(eng)")
print(f"✓ Got benchmark with {len(benchmark)} tasks")

# Test 3: Get a single task
task_name = "STS12"  # Simple task
tasks_list = [t for t in benchmark if t.metadata.name == task_name]
if not tasks_list:
    print(f"❌ Task '{task_name}' not found")
    sys.exit(1)
print(f"✓ Found task: {task_name}")

# Test 4: Create evaluator
evaluator = MTEB(tasks=tasks_list)
print("✓ Created MTEB evaluator")

# Test 5: Check evaluator.run signature
import inspect
sig = inspect.signature(evaluator.run)
print(f"✓ evaluator.run signature: {sig}")

# Test 6: Check MTEBResults
from mteb import MTEBResults
if hasattr(MTEBResults, 'to_disk'):
    print("✓ MTEBResults.to_disk exists")
else:
    print("❌ MTEBResults.to_disk does NOT exist")
    sys.exit(1)

print()
print("=" * 80)
print("✅ All API checks passed!")
print("=" * 80)
print()
print("Note: To test actual evaluation, you would need:")
print("  1. A model/embedder")
print("  2. GPU resources")
print("  3. More time")
print()
print("The code should work correctly now.")
