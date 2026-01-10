# Checkpoint Configuration Guide

## Overview

This project evaluates checkpoint-layer pairs (t, ℓ) as candidate embedding models, creating a **time-layer plane search** that captures how representations evolve during training.

### Key Concept

- **Layer-by-Layer paper**: Only evaluates layers of FINAL checkpoint → `{final} × L`
- **This project**: Evaluates layers AND training checkpoints → `T × L`
- **Novel contribution**: Captures training dynamics across both checkpoint and layer dimensions

## Current Implementation Status

### ✅ Code Design

The code is correctly designed for checkpoint-layer pairs:
- Bandit treats each `(checkpoint, layer)` as a separate arm
- Metrics computed per `(checkpoint, layer)` pair
- Search space: `all_arms = [(revision, layer) for revision in cfg.revisions for layer in layers]`

### Configuration

**Current default**: `revisions: ["main"]` (single checkpoint)
- Result: Only evaluating `{main} × L`, not `T × L`
- This is identical to layer-by-layer paper (except using bandit instead of brute-force)

**To enable full time-layer search**: Add multiple checkpoints to the `revisions` list

## Finding Available Pythia Checkpoints

### Method 1: Check HuggingFace Model Pages

1. Visit: https://huggingface.co/EleutherAI/pythia-410m
2. Look for "Files and versions" tab
3. Check for checkpoint directories or revision tags

### Method 2: Use HuggingFace Hub API

```python
from huggingface_hub import list_revisions
revisions = list_revisions('EleutherAI/pythia-410m')
print(revisions)
```

### Method 3: Use Provided Script

```bash
# List available checkpoints for all model sizes
python scripts/list_available_checkpoints.py --model-size all

# List for specific model size
python scripts/list_available_checkpoints.py --model-size 410m --output checkpoints_410m.json

# Run via SLURM
sbatch slurm/check_checkpoints.sbatch
```

See `RUN_CHECKPOINT_CHECK.md` for detailed instructions.

### Common Pythia Checkpoint Patterns

Pythia models typically have checkpoints at:
- `step-{N}` format (e.g., `step1000`, `step2000`, `step10000`) - **Note: no hyphen in actual format**
- `step{NUMBER}` where NUMBER is training step (e.g., `step1000`, `step2000`)
- `main` (final checkpoint)

**IMPORTANT**: The actual checkpoint format uses `step{NUMBER}` (no hyphen), not `step-{NUMBER}`.

## How to Enable Multiple Checkpoints

### Step 1: Update Config

Edit `configs/mteb_layersweep.yaml`:

```yaml
hf:
  org: "EleutherAI"
  model_family: "Pythia"
  model_sizes: ["14m", "70m", "410m"]
  # Add multiple checkpoints here:
  revisions: ["step1000", "step2000", "step5000", "step10000", "main"]
```

**Recommended checkpoints for time-layer plane search:**
- Early training: `step1000`, `step2000`
- Mid training: `step5000`, `step10000`, `step20000`
- Late training: `step40000`, `step80000` (if available)
- Final: `main`

### Step 2: Re-run Metrics Computation

The code will automatically:
1. Compute metrics for all (checkpoint, layer) pairs
2. Create arms: `(step1000, 0)`, `(step1000, 1)`, ..., `(main, L-1)`
3. Bandit will select from this full search space

### Step 3: Verify Configuration

After enabling multiple checkpoints, verify:

```bash
# Check that metrics exist for all checkpoints
find runs/{run_id}/cache/metrics -type d -name "step*" | wc -l

# Check bandit arms include checkpoints
python3 -c "import json; s=json.load(open('runs/{run_id}/bandit_state.json')); print([a for a in s['trajectory'] if 'step' in str(a['arm'][0])])"
```

## Expected Results

With multiple checkpoints enabled, you'll see:

1. **Metrics Structure**:
   ```
   cache/metrics/410m/step1000/layer_000.json
   cache/metrics/410m/step1000/layer_001.json
   ...
   cache/metrics/410m/step2000/layer_000.json
   ...
   cache/metrics/410m/main/layer_000.json
   ```

2. **Bandit Selection**:
   - Will select across both checkpoint and layer dimensions
   - Can discover that early checkpoints + middle layers perform best
   - Or that late checkpoints + early layers are optimal

3. **Insights**:
   - How training stage affects layer performance
   - Optimal checkpoint-layer combinations
   - True time-layer plane search results

## Checkpoint Format Reference

See `PYTHIA_154_CHECKPOINTS.md` for a complete list of available checkpoints for Pythia models (155 checkpoints per model).

## Summary

- **Original idea**: ✅ Checkpoint-layer pairs (t, ℓ)
- **Code implementation**: ✅ Ready for multiple checkpoints
- **Default config**: ❌ Only "main" checkpoint
- **Solution**: Add multiple checkpoints to `revisions` list
- **Result**: True T × L time-layer plane search
