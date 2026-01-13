"""Command line entry points.

Usage:
  python -m layer_time.cli mteb-layersweep --config configs/mteb_layersweep.yaml
  python -m layer_time.cli mteb-layersweep --config ... --run-id <existing_run_id>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError

from layer_time.constants import TASK_PRESETS
from layer_time.env_utils import capture_environment
from layer_time.mteb_runner import SweepConfig, run_mteb_layer_sweep, set_global_seed
from layer_time.mteb_bandit_runner import BanditSweepConfig, run_mteb_bandit_workflow


class RunSection(BaseModel):
    run_id: Optional[str] = None
    output_root: str = "runs"
    seed: int = 42


class HFSection(BaseModel):
    org: str = "EleutherAI"
    model_family: str = "Pythia"
    model_sizes: List[str] = Field(default_factory=lambda: ["14m", "70m", "410m"])
    revisions: List[str] = Field(default_factory=lambda: ["main"])


class EmbeddingSection(BaseModel):
    pooling: str = "mean"
    normalize: bool = True
    max_length: int = 2048  # Match paper's max_sample_length for causal models
    batch_size: int = 64
    device: str = "auto"
    dtype: str = "auto"
    layers: str = "all"


class MTEBSection(BaseModel):
    tasks_preset: Optional[str] = "layer_by_layer_32"
    tasks: Optional[List[str]] = None


class ExecutionSection(BaseModel):
    fail_fast: bool = False
    resume: bool = True


class BanditSection(BaseModel):
    enabled: bool = False
    algorithm: str = "linUCB"
    alpha: float = 1.0
    budget: int = 100
    baseline_checkpoint: str = "main"


class MetricsSection(BaseModel):
    corpus_max_examples_per_task: Optional[int] = None
    corpus_cache_path: Optional[str] = None


class RewardsSection(BaseModel):
    aggregation_method: str = "mean"  # mean, harmonic_mean, robust_mean
    baseline_method: str = "mean"


class LayersweepConfig(BaseModel):
    run: RunSection
    hf: HFSection
    embedding: EmbeddingSection
    mteb: MTEBSection
    execution: ExecutionSection
    bandit: Optional[BanditSection] = None
    metrics: Optional[MetricsSection] = None
    rewards: Optional[RewardsSection] = None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_tasks(mteb: MTEBSection) -> List[str]:
    if mteb.tasks is not None:
        return list(mteb.tasks)
    if mteb.tasks_preset is None:
        raise ValueError("Either mteb.tasks or mteb.tasks_preset must be provided.")
    if mteb.tasks_preset not in TASK_PRESETS:
        raise ValueError(f"Unknown tasks_preset: {mteb.tasks_preset}. Available: {list(TASK_PRESETS)}")
    return TASK_PRESETS[mteb.tasks_preset]


def cmd_mteb_layersweep(args: argparse.Namespace) -> None:
    cfg_raw = _load_yaml(Path(args.config))
    try:
        cfg = LayersweepConfig(**cfg_raw)
    except ValidationError as e:
        raise SystemExit(f"Invalid config: {e}") from e

    # Run id resolution (CLI overrides YAML)
    run_id = args.run_id or cfg.run.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg.run.output_root)
    run_dir = output_root / run_id

    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot config
    (run_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(cfg_raw, sort_keys=False), encoding="utf-8"
    )

    # Record env + seed
    capture_environment(run_dir)
    set_global_seed(cfg.run.seed)

    tasks = _resolve_tasks(cfg.mteb)

    # Check if bandit mode is enabled
    bandit_enabled = cfg.bandit is not None and cfg.bandit.enabled
    
    if bandit_enabled:
        # Run bandit workflow
        metrics_cfg = cfg.metrics or MetricsSection()
        rewards_cfg = cfg.rewards or RewardsSection()
        bandit_cfg = cfg.bandit
        
        corpus_cache_path = None
        if metrics_cfg.corpus_cache_path:
            corpus_cache_path = Path(metrics_cfg.corpus_cache_path)
        
        bandit_sweep_cfg = BanditSweepConfig(
            run_id=run_id,
            output_root=output_root,
            hf_org=cfg.hf.org,
            model_family=cfg.hf.model_family,
            model_sizes=cfg.hf.model_sizes,
            revisions=cfg.hf.revisions,
            pooling=cfg.embedding.pooling,
            normalize=cfg.embedding.normalize,
            max_length=cfg.embedding.max_length,
            batch_size=cfg.embedding.batch_size,
            device=cfg.embedding.device,
            dtype=cfg.embedding.dtype,
            layers=cfg.embedding.layers,
            tasks=tasks,
            bandit_alpha=bandit_cfg.alpha,
            bandit_budget=bandit_cfg.budget,
            baseline_checkpoint=bandit_cfg.baseline_checkpoint,
            corpus_max_examples_per_task=metrics_cfg.corpus_max_examples_per_task,
            corpus_cache_path=corpus_cache_path,
            reward_aggregation_method=rewards_cfg.aggregation_method,
            reward_baseline_method=rewards_cfg.baseline_method,
            resume=cfg.execution.resume,
            fail_fast=cfg.execution.fail_fast,
            seed=cfg.run.seed,
        )
        
        run_mteb_bandit_workflow(bandit_sweep_cfg)
    else:
        # Run brute-force sweep (original behavior)
        sweep_cfg = SweepConfig(
            run_id=run_id,
            output_root=output_root,
            hf_org=cfg.hf.org,
            model_family=cfg.hf.model_family,
            model_sizes=cfg.hf.model_sizes,
            revisions=cfg.hf.revisions,
            pooling=cfg.embedding.pooling,
            normalize=cfg.embedding.normalize,
            max_length=cfg.embedding.max_length,
            batch_size=cfg.embedding.batch_size,
            device=cfg.embedding.device,
            dtype=cfg.embedding.dtype,
            layers=cfg.embedding.layers,
            tasks=tasks,
            resume=cfg.execution.resume,
            fail_fast=cfg.execution.fail_fast,
            seed=cfg.run.seed,
        )
        
        run_mteb_layer_sweep(sweep_cfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="layer_time")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("mteb-layersweep", help="Run MTEB layer sweep with resume support.")
    p_ls.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    p_ls.add_argument("--run-id", type=str, default=None, help="Optional run id to resume.")
    p_ls.set_defaults(func=cmd_mteb_layersweep)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
