# Layer-by-Layer Pipeline

This pipeline implements the exact methodology from the "Layer by Layer" paper (Skean et al., 2025) and the [information_flow repository](https://github.com/OFSkean/information_flow).

## Experiments

### Experiment 1: Main Checkpoint, Final 9 Layers
- **Model**: Pythia 410m
- **Checkpoint**: main (final checkpoint)
- **Layers**: 15-23 (final 9 layers)
- **Tasks**: All 32 MTEB tasks
- **Purpose**: Evaluate layer-wise performance on final checkpoint

### Experiment 2: Final 4 Layers, Final 50 Checkpoints
- **Model**: Pythia 410m
- **Checkpoints**: Final 50 checkpoints (step94000 through step143000)
- **Layers**: 20-23 (final 4 layers)
- **Tasks**: All 32 MTEB tasks
- **Purpose**: Evaluate checkpoint-wise performance on final layers

## Running the Pipeline

The experiments run **sequentially** (Experiment 2 starts after Experiment 1 completes):

```bash
# Run both experiments in order
./pipeline_layer_by_layer/scripts/run_pipeline.sh

# Or run individually:
# Experiment 1 only
./pipeline_layer_by_layer/scripts/run_experiment1.sh

# Experiment 2 only (after Experiment 1 completes)
./pipeline_layer_by_layer/scripts/run_experiment2.sh
```

## Results

Results are saved in:
- Experiment 1: `pipeline_layer_by_layer/results/exp1_main_final9layers/`
- Experiment 2: `pipeline_layer_by_layer/results/exp2_final50checkpoints_final4layers/`

Each experiment includes:
- MTEB task scores (per layer/checkpoint)
- Representation metrics (entropy, curvature, effective rank)
- Aggregated results CSV
- Detailed logs

## Methodology

This pipeline follows the exact methodology from:
- Paper: "Layer by Layer: Uncovering Hidden Representations in Language Models" (Skean et al., 2025)
- Repository: https://github.com/OFSkean/information_flow
- Uses the same 32 MTEB tasks from Table 1
- Calculates the same representation metrics
