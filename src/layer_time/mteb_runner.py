"""
Resumable MTEB runner for layer sweeps.

Design:
- Load the HF model once per (model_id, revision).
- Sweep layers by changing `embedder.layer_index`.
- For each (layer, task), run MTEB for that single task and write a `done.json`.
- Restarting the run will skip any (layer, task) that already has done.json.

MTEB v2 changes:
- Use mteb.get_tasks(...) + mteb.evaluate(...)
- Use ResultCache + OverwriteStrategy to avoid deprecated evaluator path.

NEW (for Slurm sharding / clean parallel output):
- Environment variables:
    LAYER_TIME_ONLY_SIZE   : if set, only run that model size (e.g. "70m" or "410m")
    LAYER_TIME_NUM_SHARDS  : total number of shards (e.g. 2 or 4)
    LAYER_TIME_SHARD_ID    : shard index in [0, num_shards-1]
- Tasks are deterministically assigned to shards (stable hash), so shards don't overlap.
- runner logs + progress CSV are written per-shard to avoid file corruption when many jobs share the same run_id.
"""

from __future__ import annotations

import json
import os
import random
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import mteb

# Be robust across MTEB version differences (v1 vs v2).
# v1.14.19 uses MTEB evaluator class, v2.x uses mteb.evaluate() function.
MTEB_VERSION = getattr(mteb, "__version__", "unknown")
USE_V1_API = MTEB_VERSION.startswith("1.")

if USE_V1_API:
    # v1.14.19 API: Use MTEB evaluator class
    try:
        from mteb import MTEB as MTEBEvaluator
    except ImportError:
        MTEBEvaluator = None
    # v1 doesn't have ResultCache or OverwriteStrategy
    ResultCache = None
    OverwriteStrategy = None
else:
    # v2.x API: Use ResultCache and OverwriteStrategy
    try:
        from mteb.results import ResultCache, OverwriteStrategy  # newer layout
    except Exception:
        try:
            from mteb.cache import ResultCache  # older layout
            from mteb.evaluate import OverwriteStrategy
        except Exception:
            ResultCache = None
            OverwriteStrategy = None

from layer_time.constants import pythia_model_id
from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.logging_utils import setup_logging


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


@dataclass(frozen=True)
class SweepConfig:
    run_id: str
    output_root: Path

    hf_org: str
    model_family: str
    model_sizes: List[str]
    revisions: List[str]

    pooling: str
    normalize: bool
    max_length: int
    batch_size: int
    device: str
    dtype: str
    layers: str  # "all" or comma-separated list

    tasks: List[str]

    resume: bool = True
    fail_fast: bool = False
    seed: int = 42


def _layer_list(embedder: HFHiddenStateEmbedder, layers_spec: str) -> List[int]:
    if layers_spec == "all":
        return list(range(embedder.num_hidden_layers))
    out: List[int] = []
    for token in layers_spec.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _safe_json_dump(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _result_to_pretty_json(result: Any) -> str:
    # MTEB uses pydantic models for results; support both v2 and v1.
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2)
    if hasattr(result, "json"):
        return result.json(indent=2)
    return json.dumps(result, indent=2, default=str)


def _task_bucket(task_name: str, n: int) -> int:
    """Deterministic stable bucket assignment (no dependence on PYTHONHASHSEED)."""
    return zlib.adler32(task_name.encode("utf-8")) % n


def _get_shard_env() -> tuple[int, int]:
    """
    Returns (num_shards, shard_id).
    Defaults to (1, 0) when env vars not set.
    """
    num_shards = int(os.environ.get("LAYER_TIME_NUM_SHARDS", "1"))
    shard_id = int(os.environ.get("LAYER_TIME_SHARD_ID", "0"))
    if num_shards < 1:
        raise ValueError(f"LAYER_TIME_NUM_SHARDS must be >= 1, got {num_shards}")
    if not (0 <= shard_id < num_shards):
        raise ValueError(
            f"LAYER_TIME_SHARD_ID must be in [0, {num_shards-1}], got {shard_id}"
        )
    return num_shards, shard_id


def run_mteb_layer_sweep(cfg: SweepConfig) -> None:
    """Run a resumable MTEB sweep across sizes/revisions/layers/tasks."""
    num_shards, shard_id = _get_shard_env()
    only_size = os.environ.get("LAYER_TIME_ONLY_SIZE", "").strip() or None

    run_dir = cfg.output_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: when multiple shards share the same run_id, they must not write the same log/progress file.
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / (
        f"runner.log" if num_shards == 1 and not only_size else f"runner_shard{shard_id:02d}.log"
    )
    logger = setup_logging(log_path)

    logger.info(
        "Starting MTEB layer sweep | run_dir=%s | num_shards=%d shard_id=%d only_size=%s",
        run_dir,
        num_shards,
        shard_id,
        only_size,
    )

    set_global_seed(cfg.seed)

    # Per-shard progress file (avoids concurrent appends corrupting a shared CSV).
    progress_path = run_dir / (
        "progress.csv" if num_shards == 1 and not only_size else f"progress_shard{shard_id:02d}.csv"
    )
    if not progress_path.exists():
        pd.DataFrame(
            columns=[
                "timestamp",
                "model_size",
                "revision",
                "layer",
                "task",
                "status",
                "elapsed_sec",
                "output_dir",
                "error",
            ]
        ).to_csv(progress_path, index=False)

    # Decide which sizes this job will run.
    model_sizes = list(cfg.model_sizes)
    if only_size:
        model_sizes = [s for s in model_sizes if s == only_size]
        if not model_sizes:
            logger.warning("No model_sizes matched only_size=%s; nothing to do.", only_size)
            return

    # Decide which tasks this shard will run.
    all_tasks = list(cfg.tasks)
    if num_shards > 1:
        shard_tasks = [t for t in all_tasks if _task_bucket(t, num_shards) == shard_id]
    else:
        shard_tasks = all_tasks

    shard_meta = {
        "run_id": cfg.run_id,
        "only_size": only_size,
        "num_shards": num_shards,
        "shard_id": shard_id,
        "model_sizes": model_sizes,
        "revisions": list(cfg.revisions),
        "num_tasks_total": len(all_tasks),
        "num_tasks_this_shard": len(shard_tasks),
        "tasks_this_shard": shard_tasks,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _safe_json_dump(shard_meta, run_dir / (f"shard_{shard_id:02d}_meta.json"))

    if not shard_tasks:
        logger.warning(
            "This shard has 0 tasks (num_shards=%d shard_id=%d). Exiting.",
            num_shards,
            shard_id,
        )
        return

    for size in model_sizes:
        model_id = pythia_model_id(size=size, org=cfg.hf_org)
        for revision in cfg.revisions:
            logger.info("Loading model | model_id=%s revision=%s", model_id, revision)

            # Load once; reuse for all layers/tasks.
            embedder = HFHiddenStateEmbedder(
                model_id=model_id,
                revision=revision,
                pooling=cfg.pooling,
                normalize=cfg.normalize,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                device=cfg.device,
                dtype=cfg.dtype,
                layer_index=0,
            )

            layers = _layer_list(embedder, cfg.layers)
            logger.info(
                "Layer sweep | num_hidden_layers=%d layers=%s",
                embedder.num_hidden_layers,
                layers,
            )

            for layer in layers:
                embedder.set_layer(layer)

                for task_name in shard_tasks:
                    out_dir = (
                        run_dir
                        / "outputs"
                        / "mteb"
                        / cfg.model_family
                        / size
                        / revision
                        / f"layer_{layer:03d}"
                        / task_name
                    )
                    out_dir.mkdir(parents=True, exist_ok=True)
                    done_path = out_dir / "done.json"

                    if cfg.resume and done_path.exists():
                        logger.info(
                            "Skip (already done) | size=%s rev=%s layer=%d task=%s",
                            size,
                            revision,
                            layer,
                            task_name,
                        )
                        continue

                    start_t = time.time()
                    status = "ok"
                    err_msg = ""

                    try:
                        logger.info(
                            "Run task | size=%s rev=%s layer=%d task=%s",
                            size,
                            revision,
                            layer,
                            task_name,
                        )

                        if USE_V1_API:
                            # v1.14.19 API: Use MTEB evaluator
                            benchmark = mteb.get_benchmark("MTEB(eng)")
                            tasks_list = [t for t in benchmark if t.metadata.name == task_name]
                            if not tasks_list:
                                raise ValueError(f"Task '{task_name}' not found in MTEB(eng) benchmark")
                            
                            evaluator = MTEBEvaluator(tasks=tasks_list)
                            
                            # Run evaluation using v1 API
                            # v1 API saves to {output_folder}/{model_name}/{model_revision}/{task_name}.json
                            # We'll use a temp location and then copy results to our desired location
                            import tempfile
                            with tempfile.TemporaryDirectory() as temp_output:
                                task_results = evaluator.run(
                                    embedder,
                                    verbosity=0,  # 0 = minimal output (suppress progress bars)
                                    output_folder=temp_output,
                                    overwrite_results=True,
                                    raise_error=True,
                                    encode_kwargs={
                                        "batch_size": cfg.batch_size,
                                        "max_length": cfg.max_length,
                                    },
                                )
                                
                                # v1 returns list of MTEBResults objects
                                if task_results and len(task_results) > 0:
                                    result = task_results[0]
                                    # Save result to our custom location
                                    result_path = out_dir / f"{task_name}.json"
                                    out_dir.mkdir(parents=True, exist_ok=True)
                                    result.to_disk(result_path)  # to_disk accepts Path object
                                else:
                                    raise ValueError(f"No results returned for task '{task_name}'")
                        else:
                            # v2.x API: Use mteb.evaluate() function
                            tasks_obj = mteb.get_tasks(tasks=[task_name])

                            # Cache isolated per task/layer dir => no cross-shard collisions.
                            cache = ResultCache(cache_path=str(out_dir))

                            result = mteb.evaluate(
                                model=embedder,
                                tasks=tasks_obj,
                                cache=cache,
                                overwrite_strategy=OverwriteStrategy.ONLY_MISSING,
                                encode_kwargs={
                                    "batch_size": cfg.batch_size,
                                    "max_length": cfg.max_length,
                                },
                                show_progress_bar=True,
                                raise_error=True,  # we catch and record below
                            )

                            (out_dir / f"{task_name}.json").write_text(
                                _result_to_pretty_json(result),
                                encoding="utf-8",
                            )

                        _safe_json_dump(
                            {
                                "status": "done",
                                "model_id": model_id,
                                "model_size": size,
                                "revision": revision,
                                "layer": layer,
                                "task": task_name,
                                "num_shards": num_shards,
                                "shard_id": shard_id,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            },
                            done_path,
                        )

                    except Exception as e:
                        status = "failed"
                        err_msg = repr(e)
                        logger.exception(
                            "Task failed | size=%s rev=%s layer=%d task=%s",
                            size,
                            revision,
                            layer,
                            task_name,
                        )
                        _safe_json_dump(
                            {
                                "status": "failed",
                                "model_id": model_id,
                                "model_size": size,
                                "revision": revision,
                                "layer": layer,
                                "task": task_name,
                                "num_shards": num_shards,
                                "shard_id": shard_id,
                                "error": err_msg,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            },
                            out_dir / "failed.json",
                        )
                        if cfg.fail_fast:
                            raise

                    elapsed = time.time() - start_t

                    row = pd.DataFrame(
                        [
                            {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "model_size": size,
                                "revision": revision,
                                "layer": layer,
                                "task": task_name,
                                "status": status,
                                "elapsed_sec": round(elapsed, 3),
                                "output_dir": str(out_dir),
                                "error": err_msg,
                            }
                        ]
                    )
                    row.to_csv(progress_path, mode="a", header=False, index=False)

            # Best-effort free GPU memory between sizes/revisions
            try:
                import torch

                del embedder
                torch.cuda.empty_cache()
            except Exception:
                pass

    logger.info("Sweep complete | shard_id=%d num_shards=%d only_size=%s", shard_id, num_shards, only_size)