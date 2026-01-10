# Interactive Shell Testing Guide

## Quick Test Commands

### Step 1: Navigate and Setup

```bash
# Navigate to project directory
cd /project/6101803/mahdiar/pythia-layer-time

# Load modules (if on compute node)
module --force purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2

# Activate virtual environment
source lbl/bin/activate

# Set up Python path
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Set up HuggingFace cache
export HF_HOME=/scratch/$USER/hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export XDG_CACHE_HOME=$HF_HOME/xdg
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
```

### Step 2: Quick Verification

```bash
# Check Python and GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Check imports
python -c "import layer_time.mteb_bandit_runner; print('✅ Imports OK')"
```

### Step 3: Test Checkpoint Loading (Optional)

Test if checkpoint format works:

```bash
python -c "
from layer_time.embedder import HFHiddenStateEmbedder
import torch

print('Testing pythia-14m @ main...')
embedder = HFHiddenStateEmbedder(
    model_id='EleutherAI/pythia-14m',
    revision='main',
    pooling='mean',
    normalize=True,
    max_length=256,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dtype='auto',
    layer_index=0,
)
print(f'✅ Loaded! Layers: {embedder.num_hidden_layers}')
"
```

### Step 4: Run Automated Test Script

The easiest way - run the automated test script:

```bash
./test_pipeline_interactive.sh
```

This script will:
- Check environment
- Load modules
- Activate virtual environment
- Verify imports
- Optionally test checkpoint loading
- Run a minimal pipeline test

### Step 5: Run Manual Test

Or run the test manually:

```bash
python -m layer_time.cli mteb-layersweep \
    --config configs/mteb_test.yaml \
    --run-id "test_run_$(date +%Y%m%d_%H%M%S)"
```

### Step 6: Monitor Progress

While the test runs, monitor it:

```bash
# In another terminal or with Ctrl+Z (bg), check progress:
tail -f runs/test_run_*/logs/bandit_runner.log

# Or check progress CSV:
watch -n 5 'tail -20 runs/test_run_*/progress.csv'
```

### Step 7: Check Results

After completion:

```bash
# Check if test completed
ls -lh runs/test_run_*/

# View progress
cat runs/test_run_*/progress.csv

# View best arm
cat runs/test_run_*/best_arm.json

# View logs
tail -100 runs/test_run_*/logs/bandit_runner.log
```

## Test Configuration Explained

The test config (`configs/mteb_test.yaml`) uses:

- **Model**: `pythia-14m` (smallest, fastest)
- **Checkpoints**: `step0`, `step1000`, `step2000`, `main` (4 checkpoints)
- **Layers**: `0, 1, 2` (first 3 layers only)
- **Tasks**: `STS12`, `STS13` (2 simple tasks)
- **Budget**: `5` evaluations (very small)
- **Corpus**: `100` examples per task (small for speed)

**Expected runtime**: 10-30 minutes on GPU

**Expected outputs**:
- `progress.csv`: 5 rows (one per evaluation)
- `best_arm.json`: Best (checkpoint, layer) pair
- `bandit_state.json`: Final bandit state
- `logs/bandit_runner.log`: Detailed logs

## Troubleshooting

### GPU Not Available

If you see "CUDA: False", you might be on a login node:

```bash
# Request interactive GPU session
srun --gres=gpu:l40s:1 --mem=120G --time=2:00:00 --pty bash

# Or use H100
srun --gres=gpu:h100:1 --mem=120G --time=2:00:00 --pty bash
```

### Import Errors

If imports fail:

```bash
# Verify virtual environment
which python  # Should point to lbl/bin/python

# Reinstall dependencies if needed
pip install -r requirements.txt  # If you have one
```

### Checkpoint Loading Errors

If checkpoint loading fails:

```bash
# Test with a checkpoint that definitely exists (main)
python -c "
from layer_time.embedder import HFHiddenStateEmbedder
import torch

embedder = HFHiddenStateEmbedder(
    model_id='EleutherAI/pythia-14m',
    revision='main',  # This should always exist
    pooling='mean',
    normalize=True,
    max_length=256,
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dtype='auto',
    layer_index=0,
)
print('✅ main checkpoint works')
"
```

If `main` works but `step1000` doesn't, that checkpoint might not exist for the 14m model (which is OK).

### Out of Memory (OOM)

If you hit OOM errors:

```bash
# Reduce batch size in test config
nano configs/mteb_test.yaml
# Change: batch_size: 32 → batch_size: 16
```

### Test Takes Too Long

If the test is taking too long:

```bash
# Reduce budget further
nano configs/mteb_test.yaml
# Change: budget: 5 → budget: 2

# Or reduce corpus size
# Change: corpus_max_examples_per_task: 100 → corpus_max_examples_per_task: 50
```

## What to Check After Test

1. **Progress CSV exists and has rows**:
   ```bash
   wc -l runs/test_run_*/progress.csv
   # Should be > 1 (header + data rows)
   ```

2. **Best arm found**:
   ```bash
   cat runs/test_run_*/best_arm.json
   # Should have checkpoint and layer
   ```

3. **No errors in logs**:
   ```bash
   grep -i error runs/test_run_*/logs/bandit_runner.log
   # Should be empty or only warnings about missing checkpoints
   ```

4. **Metrics computed**:
   ```bash
   ls runs/test_run_*/cache/metrics/
   # Should have metric files
   ```

## Next Steps After Successful Test

Once the test works:

1. **Update main config** with your desired settings
2. **Submit batch job** using SLURM scripts
3. **Monitor** the batch job

```bash
# Edit main config
nano configs/mteb_layersweep.yaml

# Submit batch job
sbatch slurm/mteb_bandit_l40s.sbatch  # For L40 GPU
# or
sbatch slurm/mteb_bandit.sbatch  # For H100 GPU
```
