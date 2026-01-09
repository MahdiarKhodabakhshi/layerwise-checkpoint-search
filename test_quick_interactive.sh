#!/bin/bash
# Quick test script for interactive node
# Tests that the bandit workflow runs correctly with GPUs
# Should complete in 5-10 minutes

set -euo pipefail

echo "=========================================="
echo "Quick Interactive Test - Bandit Pipeline"
echo "=========================================="
echo ""

cd /project/6101803/mahdiar/pythia-layer-time

# Step 1: Check GPU availability
echo "Step 1: Checking GPU availability..."
echo "----------------------------------------"
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPUs: {torch.cuda.device_count()}'); [print(f'   GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null; then
    echo "❌ Error: CUDA not available or PyTorch not installed"
    echo "Make sure you're on a compute node with GPUs allocated"
    exit 1
fi

# Step 2: Check environment
echo ""
echo "Step 2: Checking environment..."
echo "----------------------------------------"
if [ ! -d "lbl/bin" ]; then
    echo "❌ Error: Virtual environment not found"
    exit 1
fi

# Activate virtual environment
source lbl/bin/activate || {
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
}

# Check pyarrow
if ! python -c "import pyarrow" 2>/dev/null; then
    echo "⚠️  pyarrow not found, installing..."
    pip install pyarrow --quiet
fi

# Set PYTHONPATH
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Step 3: Verify imports
echo ""
echo "Step 3: Verifying imports..."
echo "----------------------------------------"
if ! python -c "from layer_time.mteb_bandit_runner import run_mteb_bandit_workflow; print('✅ Bandit runner imported')" 2>/dev/null; then
    echo "❌ Error: Failed to import bandit runner"
    exit 1
fi

# Step 4: Check test config exists
echo ""
echo "Step 4: Checking test configuration..."
echo "----------------------------------------"
if [ ! -f "configs/mteb_layersweep_test.yaml" ]; then
    echo "❌ Error: Test config not found"
    exit 1
fi
echo "✅ Test config found"

# Step 5: Run quick test
echo ""
echo "Step 5: Running quick test..."
echo "----------------------------------------"
echo "This will run a minimal test:"
echo "  - Model: 14m (smallest)"
echo "  - Checkpoints: step0, main (2 checkpoints)"
echo "  - Layers: 0, 1 (first 2 layers)"
echo "  - Budget: 2 evaluations"
echo "  - Corpus: 50 examples per task (very small)"
echo ""
echo "Expected time: 5-10 minutes"
echo ""

# Set HuggingFace cache
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# Generate run ID
RUN_ID="test_interactive_$(date +%Y%m%d_%H%M%S)"
echo "Run ID: $RUN_ID"
echo ""

echo "Starting test workflow..."
python -m layer_time.cli mteb-layersweep \
  --config configs/mteb_layersweep_test.yaml \
  --run-id "$RUN_ID"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Test completed successfully!"
    echo ""
    echo "Results are in: runs/$RUN_ID"
    echo ""
    echo "To check results:"
    echo "  cat runs/$RUN_ID/bandit_results.json"
    echo "  cat runs/$RUN_ID/bandit_progress.csv"
else
    echo "❌ Test failed with exit code $EXIT_CODE"
    echo ""
    echo "Check logs:"
    echo "  tail -50 runs/$RUN_ID/logs/bandit_runner.log"
fi
echo "=========================================="

exit $EXIT_CODE
