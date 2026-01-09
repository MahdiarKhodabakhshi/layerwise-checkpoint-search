#!/bin/bash
# Interactive shell test script for bandit pipeline
# Run this in your interactive shell to test the pipeline step by step

set -euo pipefail

echo "=========================================="
echo "Testing Bandit Pipeline - Step by Step"
echo "=========================================="
echo ""

# Step 1: Check environment
echo "Step 1: Checking environment..."
echo "----------------------------------------"

# Check if we're in the right directory
if [ ! -f "src/layer_time/cli.py" ]; then
    echo "❌ Error: Not in pythia-layer-time directory"
    echo "Please cd to /project/6101803/mahdiar/pythia-layer-time"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "lbl/bin" ]; then
    echo "❌ Error: Virtual environment not found at lbl/bin"
    echo "Please set up the virtual environment first"
    exit 1
fi

echo "✅ Directory structure OK"

# Step 2: Load modules (if on compute node)
echo ""
echo "Step 2: Loading modules..."
echo "----------------------------------------"

if command -v module &> /dev/null; then
    module --force purge || true
    module load StdEnv/2023 || true
    module load python/3.11 || true
    module load cuda/12.2 || true
    module load arrow/21.0.0 2>/dev/null || echo "⚠️  Note: arrow/21.0.0 module not available (will install pyarrow instead)"
    echo "✅ Modules loaded"
else
    echo "⚠️  Module command not available (might be on login node)"
fi

# Step 3: Activate virtual environment (MUST be done before pip install)
echo ""
echo "Step 3: Activating virtual environment..."
echo "----------------------------------------"

source lbl/bin/activate || {
    echo "❌ Error: Failed to activate virtual environment"
    exit 1
}

echo "✅ Virtual environment activated: $(which python)"

# Step 3b: Check and install pyarrow if needed (required for MTEB)
echo ""
echo "Step 3b: Checking for pyarrow..."
echo "----------------------------------------"

if python -c "import pyarrow" 2>/dev/null; then
    echo "✅ pyarrow is available"
else
    echo "⚠️  pyarrow not found, installing in virtual environment..."
    pip install --quiet pyarrow || {
        echo ""
        echo "❌ Failed to install pyarrow"
        echo ""
        echo "Please install manually:"
        echo "  pip install pyarrow"
        echo ""
        echo "Or if on compute node, try:"
        echo "  module load arrow/21.0.0"
        echo "  source lbl/bin/activate"
        echo "  python -c 'import pyarrow'  # Verify it works"
        exit 1
    }
    echo "✅ pyarrow installed successfully"
fi

# Step 4: Set up Python path
echo ""
echo "Step 4: Setting up Python path..."
echo "----------------------------------------"

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

echo "✅ PYTHONPATH=$PYTHONPATH"

# Step 5: Set up HuggingFace cache
echo ""
echo "Step 5: Setting up HuggingFace cache..."
echo "----------------------------------------"

export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
echo "✅ HuggingFace cache: $HF_HOME"

# Step 6: Verify Python imports
echo ""
echo "Step 6: Verifying Python imports..."
echo "----------------------------------------"

python -c "
import sys
print(f'Python version: {sys.version}')
print('Testing imports...')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch error: {e}')
    sys.exit(1)

try:
    import pyarrow
    print(f'✅ pyarrow: {pyarrow.__version__}')
except ImportError:
    print('❌ pyarrow not found!')
    print('   This should have been caught earlier. Try: pip install pyarrow')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  pyarrow warning: {e}')

try:
    import mteb
    print(f'✅ MTEB: {mteb.__version__ if hasattr(mteb, \"__version__\") else \"installed\"}')
except Exception as e:
    print(f'❌ MTEB error: {e}')
    print('   This might be due to missing pyarrow. Try: pip install pyarrow')
    sys.exit(1)

try:
    import layer_time.mteb_bandit_runner
    print('✅ layer_time.mteb_bandit_runner imported')
except Exception as e:
    print(f'❌ layer_time.mteb_bandit_runner error: {e}')
    sys.exit(1)

try:
    from layer_time.embedder import HFHiddenStateEmbedder
    print('✅ HFHiddenStateEmbedder imported')
except Exception as e:
    print(f'❌ HFHiddenStateEmbedder error: {e}')
    sys.exit(1)

print('')
print('✅ All imports successful!')
" || {
    echo "❌ Import check failed"
    exit 1
}

# Step 7: Test checkpoint loading and verify layer/checkpoint counts
echo ""
echo "Step 7: Testing checkpoint loading and verifying layer/checkpoint counts..."
echo "----------------------------------------"
echo "This will test checkpoint loading and check the number of layers and"
echo "available checkpoints for each model size (14m, 70m, 410m)."
echo ""

read -p "Do you want to test checkpoint loading and layer/checkpoint counts? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Testing checkpoint loading and verifying model configurations..."
    python -c "
import sys
sys.path.insert(0, 'src')

from layer_time.embedder import HFHiddenStateEmbedder
import torch

# Model sizes to test
model_sizes = ['14m', '70m', '410m']
model_configs = {
    '14m': {'expected_layers': 6},
    '70m': {'expected_layers': 6},
    '410m': {'expected_layers': 24},
}

# Test checkpoints (start with main, then test step checkpoints)
test_checkpoints = ['main', 'step1000', 'step2000', 'step5000', 'step10000']

print('=' * 70)
print('Testing Checkpoint Loading and Layer/Checkpoint Counts')
print('=' * 70)
print()

for model_size in model_sizes:
    model_id = f'EleutherAI/pythia-{model_size}'
    expected_layers = model_configs[model_size]['expected_layers']
    
    print(f'Model: pythia-{model_size}')
    print('-' * 70)
    
    # Test main checkpoint first (should always exist)
    try:
        print(f'  Testing @ main...', end=' ', flush=True)
        embedder = HFHiddenStateEmbedder(
            model_id=model_id,
            revision='main',
            pooling='mean',
            normalize=True,
            max_length=256,
            batch_size=32,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            dtype='auto',
            layer_index=0,
        )
        num_layers = embedder.num_hidden_layers
        hidden_size = embedder.embedding_dim
        print(f'✅ Loaded successfully')
        print(f'     Number of layers: {num_layers} (expected: {expected_layers})')
        print(f'     Hidden size: {hidden_size}')
        
        if num_layers != expected_layers:
            print(f'     ⚠️  WARNING: Expected {expected_layers} layers but got {num_layers}')
        
        # Test accessing each layer
        print(f'     Testing layer access: ', end='', flush=True)
        for layer_idx in range(min(3, num_layers)):  # Test first 3 layers
            embedder.set_layer(layer_idx)
            if layer_idx == 0:
                print(f'layer{layer_idx}', end='', flush=True)
            else:
                print(f', layer{layer_idx}', end='', flush=True)
        if num_layers > 3:
            print(f', ... layer{num_layers-1}', end='', flush=True)
        print(' ✅')
        
    except Exception as e:
        print(f'❌ Failed to load main checkpoint: {e}')
        print(f'   Skipping further tests for {model_size}')
        print()
        continue
    
    # Test step checkpoints
    print(f'  Testing step checkpoints: ', end='', flush=True)
    available_checkpoints = ['main']
    
    for checkpoint in test_checkpoints:
        if checkpoint == 'main':
            continue  # Already tested
        
        try:
            embedder_test = HFHiddenStateEmbedder(
                model_id=model_id,
                revision=checkpoint,
                pooling='mean',
                normalize=True,
                max_length=256,
                batch_size=32,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dtype='auto',
                layer_index=0,
            )
            available_checkpoints.append(checkpoint)
            print(f'{checkpoint} ✅', end=' ', flush=True)
            del embedder_test
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f'{checkpoint} ❌', end=' ', flush=True)
            # Don't fail - some checkpoints may not exist
    
    print()
    print(f'  Summary: {len(available_checkpoints)}/{len(test_checkpoints)} checkpoints available')
    available_str = ", ".join(available_checkpoints)
    print(f'           Available: {available_str}')
    print()
    
    # Clean up
    del embedder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print('=' * 70)
print('✅ Checkpoint and layer verification complete!')
print('=' * 70)
" || {
    echo "⚠️  Checkpoint loading test had issues"
    echo "   This might be OK if some checkpoints don't exist for certain models"
}
fi

# Step 8: Run minimal pipeline test
echo ""
echo "Step 8: Ready to run pipeline test"
echo "----------------------------------------"
echo ""
echo "The test configuration uses:"
echo "  - Model: pythia-14m (smallest)"
echo "  - Checkpoints: step0, step1000, step2000, main (4 checkpoints)"
echo "  - Layers: 0, 1, 2 (first 3 layers)"
echo "  - Tasks: STS12, STS13 (2 simple tasks)"
echo "  - Budget: 5 evaluations (very small)"
echo "  - Corpus: 100 examples per task (small)"
echo ""
echo "This should complete in 10-30 minutes depending on GPU."
echo ""

read -p "Ready to run the test? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "=========================================="
    echo "Running pipeline test..."
    echo "=========================================="
    echo ""
    
    python -m layer_time.cli mteb-layersweep \
        --config configs/mteb_test.yaml \
        --run-id "test_run_interactive"
    
    echo ""
    echo "=========================================="
    echo "Test completed!"
    echo "=========================================="
    echo ""
    echo "Check results:"
    echo "  - Progress: cat runs/test_run_interactive/progress.csv"
    echo "  - Logs: cat runs/test_run_interactive/logs/bandit_runner.log"
    echo "  - Best arm: cat runs/test_run_interactive/best_arm.json"
    echo ""
else
    echo "Test cancelled. You can run it manually with:"
    echo "  python -m layer_time.cli mteb-layersweep --config configs/mteb_test.yaml --run-id test_run_interactive"
fi
