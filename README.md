# Layer by Layer, Step by Step: Adaptive Checkpoint Search for Representation Dynamics

This repository provides a **professional, resumable experiment harness** to reproduce the
**Layer-by-Layer** MTEB evaluation style: treat each transformer block's hidden state as an embedding,
evaluate **all layers** across the **32 MTEB tasks used in the paper**, and store **per-task, per-layer**
artifacts in a run directory that can be resumed after interruption.

## What this implements

- Model family: **Pythia** (EleutherAI)
- Model sizes: configurable (default: 14m, 70m, 410m)
- Tasks: the **32 MTEB tasks listed in Table 1** of the paper (see `src/layer_time/constants.py`)
- Resume logic: each atomic unit is **(model_size, revision, layer, task)** and writes a `done.json`.
  If the run is restarted, completed units are skipped.

## 1) Environment setup

### Option A: `venv` (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Option B: conda
```bash
conda create -n pythia-mteb python=3.10 -y
conda activate pythia-mteb
pip install -r requirements.txt
pip install -e .
```

## 2) Configure HuggingFace caches (recommended)

Large model/task downloads are cached. You can keep them inside the project:

```bash
export HF_HOME="$(pwd)/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

## 3) Run a full layer sweep over the 32 Layer-by-Layer tasks

1. Edit `configs/mteb_layersweep.yaml` if needed.
2. Run:

```bash
python -m layer_time.cli mteb-layersweep --config configs/mteb_layersweep.yaml
```

This creates `runs/<run_id>/` with:
- `config.snapshot.yaml`
- `env.json`, `pip_freeze.txt`
- `logs/runner.log`
- `outputs/mteb/...` (MTEB outputs + our `done.json` markers)
- `progress.csv` (one row per attempted unit)

### Resume after interruption

Re-run the same command with the same `--run-id`:

```bash
python -m layer_time.cli mteb-layersweep --config configs/mteb_layersweep.yaml --run-id <run_id>
```

Completed (size, revision, layer, task) units will be skipped.

## 4) Aggregate results into a single table

After a run finishes (or partially finishes), collect results into a CSV:

```bash
python -m layer_time.analysis.collect_results --run-dir runs/<run_id> --out runs/<run_id>/summary.csv
```

Notes:
- MTEB output formats can change across versions. The collector is conservative and tries multiple
  extraction patterns; you may need to customize `collect_results.py` if your MTEB version writes
  different JSON keys.

## Tips

- Start with `model_sizes: ["14m"]` until you're confident the pipeline runs.
- For GPU memory: reduce `batch_size` and/or set `dtype: "float16"` in the config.
- Use `max_length` conservatively (e.g., 256) for faster runs; you can increase later.

## Documentation

### Getting Started
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide with exact steps to run
- **[SLURM_RUN_GUIDE.md](SLURM_RUN_GUIDE.md)** - Complete guide for running on SLURM
- **[CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md)** - Guide for configuring multiple checkpoints

### Workflow & Monitoring
- **[FULL_WORKFLOW_SUMMARY.md](FULL_WORKFLOW_SUMMARY.md)** - Complete workflow implementation documentation
- **[MONITORING_GUIDE.md](MONITORING_GUIDE.md)** - Guide for monitoring workflow progress

### Reference Documentation
- **[docs/reference/](docs/reference/)** - Reference materials:
  - `GPU_RECOMMENDATIONS.md` - GPU selection recommendations
  - `MULTI_NODE_SETUP.md` - Multi-node setup guide
  - `TIME_LIMIT_EXPLANATION.md` - SLURM time limit guidelines
  - `MODELS_METRICS_TASKS.md` - Models, metrics, and tasks overview
  - `PYTHIA_154_CHECKPOINTS.md` - Available Pythia checkpoints
  - `INTERACTIVE_TEST_GUIDE.md` - Interactive testing guide
  - `RUN_CHECKPOINT_CHECK.md` - Checkpoint verification script guide
  - `QUICK_FIX_PYARROW.md` - Quick fix for pyarrow issues
