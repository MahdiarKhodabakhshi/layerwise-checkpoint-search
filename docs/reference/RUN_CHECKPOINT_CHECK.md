# Running Checkpoint Check Script

## Quick Run (All Model Sizes)

```bash
cd /project/6101803/mahdiar/pythia-layer-time
sbatch slurm/check_checkpoints.sbatch
```

## Check Specific Model Size

```bash
cd /project/6101803/mahdiar/pythia-layer-time
MODEL_SIZES="14m" sbatch slurm/check_checkpoints.sbatch
```

## Check Multiple Specific Sizes

```bash
cd /project/6101803/mahdiar/pythia-layer-time
MODEL_SIZES="14m 70m 410m" sbatch slurm/check_checkpoints.sbatch
```

## Manual Run (Interactive Session)

If you have an interactive session or want to run locally:

```bash
cd /project/6101803/mahdiar/pythia-layer-time
source lbl/bin/activate
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Check single model
python scripts/list_available_checkpoints.py --model-size 14m

# Check all model sizes
python scripts/list_available_checkpoints.py --model-size all

# Save to file
python scripts/list_available_checkpoints.py --model-size 14m --output checkpoints_14m.json --format json
```

## View Results

After the job completes:

```bash
# View logs
cat /scratch/mahdiar/slurm_logs/check_checkpoints_*.log

# View checkpoint files
cat checkpoints_14m.json
cat checkpoints_70m.json
cat checkpoints_410m.json
```

## Output Files

The script will create:
- `checkpoints_14m.json` - List of available checkpoints for 14m model
- `checkpoints_70m.json` - List of available checkpoints for 70m model
- `checkpoints_410m.json` - List of available checkpoints for 410m model

Each file contains a JSON object with model size as key and list of checkpoint names as value.
