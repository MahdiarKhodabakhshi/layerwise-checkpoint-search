#!/bin/bash
# Quick script to run the end-to-end test on an interactive node

set -euo pipefail

echo "=========================================="
echo "End-to-End Pipeline Test"
echo "=========================================="
echo ""

# Check if we're on a compute node
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "⚠️  WARNING: Not on a SLURM job. Make sure you're on an interactive node."
    echo "   Request one with: salloc --account=aip-btaati --gres=gpu:l40s:1 --cpus-per-task=8 --mem=120G --time=1:00:00"
    echo ""
fi

# Load modules
echo "Loading modules..."
module --force purge
module load StdEnv/2023
module load gcc
module load cuda/12.2
module load python/3.11
module load arrow/21.0.0

# Activate environment
echo "Activating environment..."
cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate

# Set environment variables
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"

# Check GPU
echo ""
echo "Checking GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check MTEB version
echo ""
echo "Checking MTEB version..."
python3 -c "import mteb; print(f'MTEB version: {mteb.__version__}')"

echo ""
echo "=========================================="
echo "Running end-to-end test..."
echo "=========================================="
echo ""

# Run the test
python3 pipeline_layer_by_layer/scripts/test_pipeline_end_to_end.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
