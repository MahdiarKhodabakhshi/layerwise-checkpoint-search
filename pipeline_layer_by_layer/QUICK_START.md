# Quick Start - Layer-by-Layer Pipeline

## Overview

This pipeline implements the exact methodology from:
- **Paper**: "Layer by Layer: Uncovering Hidden Representations in Language Models" (Skean et al., 2025)
- **Repository**: https://github.com/OFSkean/information_flow

## Experiments

### Experiment 1: Main Checkpoint, Final 9 Layers
- **Model**: Pythia 410m
- **Checkpoint**: `main` (final checkpoint)
- **Layers**: 15-23 (final 9 layers)
- **Tasks**: All 32 MTEB tasks from Table 1
- **Evaluations**: 1 × 9 × 32 = 288

### Experiment 2: Final 50 Checkpoints, Final 4 Layers
- **Model**: Pythia 410m
- **Checkpoints**: step94000 through step143000 (50 checkpoints)
- **Layers**: 20-23 (final 4 layers)
- **Tasks**: All 32 MTEB tasks from Table 1
- **Evaluations**: 50 × 4 × 32 = 6,400

## Running the Pipeline

### Option 1: Run Both Experiments Sequentially (Recommended)

```bash
cd /project/6101803/mahdiar/pythia-layer-time
./pipeline_layer_by_layer/scripts/run_pipeline.sh
```

This will:
1. Submit Experiment 1
2. Submit Experiment 2 with dependency (starts after Experiment 1 completes)

### Option 2: Run Experiments Individually

```bash
# Experiment 1 only
sbatch pipeline_layer_by_layer/slurm/exp1_main_final9layers.sbatch

# Experiment 2 only (after Experiment 1 completes)
sbatch pipeline_layer_by_layer/slurm/exp2_final50checkpoints_final4layers.sbatch
```

## Monitoring

```bash
# Check job status
squeue -u mahdiar | grep layer-by-layer

# View logs
tail -f /scratch/mahdiar/slurm_logs/layer-by-layer-exp*.log

# Check progress
ls -lh /scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer/*/progress.csv
```

## Results

Results are saved to:
```
/scratch/mahdiar/pythia-layer-time-runs/pipeline_layer_by_layer/
├── exp1_main_final9layers/
│   ├── outputs/mteb/... (MTEB results)
│   ├── progress.csv
│   └── cache/representation_metrics.csv (if computed)
└── exp2_final50checkpoints_final4layers/
    ├── outputs/mteb/... (MTEB results)
    ├── progress.csv
    └── cache/representation_metrics.csv (if computed)
```

## Collecting Results

After experiments complete, collect and aggregate results:

```bash
python pipeline_layer_by_layer/scripts/collect_results.py
```

This creates summary CSV files with all metrics.

## Computing Representation Metrics

To compute representation metrics (entropy, curvature, effective rank) matching the GitHub repo:

```bash
# For Experiment 1
python pipeline_layer_by_layer/scripts/compute_representation_metrics.py \
  pipeline_layer_by_layer/configs/exp1_main_final9layers.yaml

# For Experiment 2
python pipeline_layer_by_layer/scripts/compute_representation_metrics.py \
  pipeline_layer_by_layer/configs/exp2_final50checkpoints_final4layers.yaml
```

## Methodology Verification

This pipeline uses:
- ✅ Same 32 MTEB tasks from Table 1 of the paper
- ✅ Same evaluation framework (MTEB)
- ✅ Same representation metrics (prompt entropy, dataset entropy, curvature, effective rank)
- ✅ Same model (Pythia 410m from EleutherAI)
- ✅ Same layer evaluation approach

The implementation follows the methodology described in:
- https://github.com/OFSkean/information_flow
- Uses `layer_time.cli mteb-layersweep` which internally uses `mteb.evaluate()` (same as the repo's MTEB-Harness.py)
