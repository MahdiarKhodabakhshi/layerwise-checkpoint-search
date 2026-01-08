#!/bin/bash
# Run both experiments in sequence
# Experiment 2 starts only after Experiment 1 completes

set -euo pipefail

cd "$(dirname "$0")/../.."

echo "=========================================="
echo "Layer-by-Layer Pipeline"
echo "=========================================="
echo ""
echo "This will run two experiments in sequence:"
echo "  1. Main checkpoint, final 9 layers"
echo "  2. Final 50 checkpoints, final 4 layers"
echo ""
echo "Experiment 2 will start only after Experiment 1 completes."
echo ""

# Submit Experiment 1 (optimized with array jobs)
echo "Submitting Experiment 1 (9 array tasks, one per layer)..."
JOB1_ID=$(sbatch pipeline_layer_by_layer/slurm/exp1_main_final9layers_array.sbatch | grep -oP '\d+')
echo "Experiment 1 submitted: Array Job ID $JOB1_ID (9 tasks, 9 concurrent)"
echo "  • Uses 9 GPUs simultaneously"
echo "  • Each task processes 1 layer × 32 tasks = 32 evaluations"
echo "  • Estimated time: ~2-3 hours"
echo ""

# Wait for Experiment 1 to complete
echo "Waiting for Experiment 1 to complete..."
echo "You can monitor with: squeue -j $JOB1_ID"
echo ""

# Submit Experiment 2 with dependency on Experiment 1 (MAXIMUM optimization)
echo "Submitting Experiment 2 (200 array tasks, one per checkpoint-layer pair)..."
echo "  • Will start after Experiment 1 completes"
echo "  • Uses up to 60 GPUs (200 array tasks, 60 concurrent)"
echo "  • Each task processes 1 checkpoint × 1 layer × 32 tasks = 32 evaluations"
echo "  • Estimated time: ~6-7 hours (FASTEST!)"
echo ""

JOB2_ID=$(sbatch --dependency=afterok:$JOB1_ID pipeline_layer_by_layer/slurm/exp2_final50checkpoints_final4layers_array_max.sbatch | grep -oP '\d+')
echo "Experiment 2 submitted: Array Job ID $JOB2_ID (depends on $JOB1_ID)"
echo "  • 200 array tasks (one per checkpoint-layer pair)"
echo "  • Max 60 concurrent (uses all available GPUs)"
echo "  • Maximum parallelization for fastest completion!"
echo ""

echo "=========================================="
echo "Pipeline submitted!"
echo "=========================================="
echo "Experiment 1: Array Job ID $JOB1_ID"
echo "  • 9 array tasks (one per layer)"
echo "  • 9 concurrent (uses 9 GPUs)"
echo "  • Estimated: ~2-3 hours"
echo ""
echo "Experiment 2: Array Job ID $JOB2_ID"
echo "  • 200 array tasks (one per checkpoint-layer pair)"
echo "  • Max 60 concurrent (uses all 60 GPUs)"
echo "  • Estimated: ~6-7 hours"
echo "  • Waits for Experiment 1 to complete"
echo ""
echo "Monitor with:"
echo "  squeue -j $JOB1_ID,$JOB2_ID"
echo "  squeue -u mahdiar | grep layer-by-layer-exp"
echo ""
echo "Results will be in:"
echo "  /scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer/"
echo ""
echo "Total estimated time: ~8-10 hours (vs ~60+ hours sequential)"
echo "Speedup: ~6-7x faster! 🚀"
