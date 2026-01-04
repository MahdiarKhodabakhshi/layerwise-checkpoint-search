"""
Bandit-based MTEB runner implementing the full workflow.

This module implements the complete bandit-based workflow:
1. Define search space (checkpoint × layer)
2. Build representation corpus
3. Extract pooled embeddings for metrics
4. Compute representation metrics
5. Use bandit algorithm to select arms
6. Evaluate selected arms and compute z-scored rewards
7. Output best arm

See WORKFLOW_ANALYSIS.md for the complete workflow description.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

from layer_time.bandit import LinUCB
from layer_time.constants import pythia_model_id
from layer_time.corpus import build_representation_corpus, load_cached_corpus
from layer_time.embedder import HFHiddenStateEmbedder
from layer_time.logging_utils import setup_logging
from layer_time.metrics import compute_representation_metrics
from layer_time.rewards import (
    aggregate_rewards,
    compute_baseline_scores,
    compute_task_reward,
)


@dataclass(frozen=True)
class BanditSweepConfig:
    """Configuration for bandit-based workflow."""
    
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
    
    # Bandit parameters
    bandit_alpha: float = 1.0
    bandit_budget: int = 100
    baseline_checkpoint: str = "main"
    
    # Metrics parameters
    corpus_max_examples_per_task: Optional[int] = None
    corpus_cache_path: Optional[Path] = None
    
    # Reward parameters
    reward_aggregation_method: str = "mean"
    reward_baseline_method: str = "mean"
    
    resume: bool = True
    fail_fast: bool = False
    seed: int = 42


def _layer_list(embedder: HFHiddenStateEmbedder, layers_spec: str) -> List[int]:
    """Parse layer specification into list of layer indices."""
    if layers_spec == "all":
        return list(range(embedder.num_hidden_layers))
    out: List[int] = []
    for token in layers_spec.split(","):
        token = token.strip()
        if token:
            out.append(int(token))
    return out


def _safe_json_dump(obj: Any, path: Path) -> None:
    """Safely dump JSON to file."""
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def _result_to_pretty_json(result: Any) -> str:
    """Convert MTEB result to pretty JSON string."""
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2)
    if hasattr(result, "json"):
        return result.json(indent=2)
    return json.dumps(result, indent=2, default=str)


def _cache_embeddings_path(run_dir: Path, checkpoint: str, layer: int, model_size: str) -> Path:
    """Get path for cached embeddings."""
    return run_dir / "cache" / "embeddings" / model_size / checkpoint / f"layer_{layer:03d}.npy"


def _cache_metrics_path(run_dir: Path, checkpoint: str, layer: int, model_size: str) -> Path:
    """Get path for cached metrics."""
    return run_dir / "cache" / "metrics" / model_size / checkpoint / f"layer_{layer:03d}.json"


def _save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    """Save embeddings to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)


def _load_embeddings(path: Path) -> Optional[np.ndarray]:
    """Load embeddings from cache."""
    if not path.exists():
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _save_metrics(metrics: Dict[str, float], path: Path) -> None:
    """Save metrics to cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _safe_json_dump(metrics, path)


def _load_metrics(path: Path) -> Optional[Dict[str, float]]:
    """Load metrics from cache."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_embeddings_for_corpus(
    embedder: HFHiddenStateEmbedder,
    corpus: List[str],
    batch_size: int,
) -> np.ndarray:
    """Extract embeddings for representation corpus."""
    return embedder.encode(corpus, batch_size=batch_size)


def _compute_metrics_for_checkpoint_layer(
    embedder: HFHiddenStateEmbedder,
    checkpoint: str,
    layer: int,
    corpus: List[str],
    model_size: str,
    run_dir: Path,
    batch_size: int,
    logger: Any,
) -> Dict[str, float]:
    """Compute metrics for a (checkpoint, layer) pair, using cache if available."""
    metrics_path = _cache_metrics_path(run_dir, checkpoint, layer, model_size)
    
    # Try to load cached metrics
    cached_metrics = _load_metrics(metrics_path)
    if cached_metrics is not None:
        logger.debug(f"Loaded cached metrics for {checkpoint} layer {layer}")
        return cached_metrics
    
    # Compute metrics
    embeddings_path = _cache_embeddings_path(run_dir, checkpoint, layer, model_size)
    
    # Try to load cached embeddings
    embeddings = _load_embeddings(embeddings_path)
    if embeddings is None:
        logger.info(f"Extracting embeddings for {checkpoint} layer {layer}")
        embeddings = _extract_embeddings_for_corpus(embedder, corpus, batch_size)
        _save_embeddings(embeddings, embeddings_path)
    
    # Compute metrics
    logger.info(f"Computing metrics for {checkpoint} layer {layer}")
    metrics = compute_representation_metrics(embeddings)
    
    # Cache metrics
    _save_metrics(metrics, metrics_path)
    
    return metrics


def _generate_all_work_items(
    cfg: BanditSweepConfig,
    logger: Any,
) -> List[Tuple[str, str, int]]:
    """
    Generate all (model_size, checkpoint, layer) work items.
    
    Returns:
        List of (model_size, checkpoint, layer) tuples
    """
    work_items: List[Tuple[str, str, int]] = []
    
    for model_size in cfg.model_sizes:
        model_id = pythia_model_id(size=model_size, org=cfg.hf_org)
        
        for revision in cfg.revisions:
            try:
                # Try to load model to get layer count
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
                
                for layer in layers:
                    work_items.append((model_size, revision, layer))
                
                # Free GPU memory
                try:
                    import torch
                    del embedder
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                    
            except Exception as e:
                # Skip unavailable checkpoints
                error_msg = str(e)
                error_type = type(e).__name__
                if "RevisionNotFoundError" in error_type or "RevisionNotFoundError" in error_msg or "not a valid git identifier" in error_msg or "404" in error_msg:
                    logger.warning(f"Skipping unavailable checkpoint: {model_size} @ {revision}")
                else:
                    logger.warning(f"Skipping checkpoint {model_size} @ {revision}: {error_type}")
                continue
    
    return work_items


def _get_distributed_env() -> Optional[Tuple[int, int, int]]:
    """
    Get distributed environment variables from SLURM.
    
    Returns:
        (worker_id, gpu_id, num_workers) or None if not in distributed mode
    """
    worker_id = os.environ.get("SLURM_PROCID")
    local_id = os.environ.get("SLURM_LOCALID")
    num_workers = os.environ.get("SLURM_NTASKS")
    
    if worker_id is None or local_id is None or num_workers is None:
        return None
    
    try:
        return (int(worker_id), int(local_id), int(num_workers))
    except (ValueError, TypeError):
        return None


def _run_distributed_phase1(
    cfg: BanditSweepConfig,
    corpus: List[str],
    run_dir: Path,
    logger: Any,
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, int], str]]:
    """
    Run Phase 1 (metrics computation) in distributed mode.
    
    Returns:
        (context_features, arm_to_model_size) - same as sequential Phase 1
    """
    from layer_time.distributed_worker import WorkQueue, WorkItem, run_distributed_worker
    
    worker_id, gpu_id, num_workers = _get_distributed_env()
    is_master = (worker_id == 0)
    
    queue_dir = run_dir / "cache" / "work_queue"
    queue = WorkQueue(queue_dir, num_workers)
    
    if is_master:
        # Master: Generate all work items and enqueue them
        logger.info(f"Master (worker 0): Generating work items for {num_workers} workers")
        work_items_list = _generate_all_work_items(cfg, logger)
        logger.info(f"Master: Generated {len(work_items_list)} work items")
        
        # Convert to WorkItem objects and enqueue
        work_items = [
            WorkItem(model_size=model_size, checkpoint=checkpoint, layer=layer)
            for model_size, checkpoint, layer in work_items_list
        ]
        queue.enqueue(work_items)
        logger.info(f"Master: Enqueued {len(work_items)} work items")
    
    # All workers (including master) process work items
    logger.info(f"Worker {worker_id}: Starting distributed worker on GPU {gpu_id}")
    
    # Workers need to load corpus from cache (master already has it in memory)
    if not is_master:
        corpus_cache_path = cfg.corpus_cache_path or (run_dir / "cache" / "representation_corpus.json")
        
        # Wait for corpus cache to be written by master (avoid race conditions)
        max_wait = 600  # seconds
        poll_interval = 5
        start_time = time.time()
        corpus = None
        
        while time.time() - start_time < max_wait:
            corpus = load_cached_corpus(corpus_cache_path)
            if corpus is not None:
                break
            logger.info(
                f"Worker {worker_id}: Corpus cache not yet available at {corpus_cache_path}, "
                f"sleeping {poll_interval}s..."
            )
            time.sleep(poll_interval)
        
        if corpus is None:
            # If corpus is still not available after waiting, log error and exit worker gracefully
            logger.error(
                f"Worker {worker_id}: Corpus cache not found after waiting at {corpus_cache_path}. "
                f"Exiting worker."
            )
            return {}, {}
        
        logger.info(f"Worker {worker_id}: Loaded corpus from cache ({len(corpus)} examples)")
    
    run_distributed_worker(
        queue_dir=queue_dir,
        worker_id=worker_id,
        gpu_id=gpu_id,
        corpus=corpus,
        run_dir=run_dir,
        batch_size=cfg.batch_size,
        cfg=cfg,
        logger=logger,
    )
    
    if is_master:
        # Master: Wait for all work to complete and collect results
        logger.info("Master: Waiting for all workers to complete...")
        
        # Poll until queue is empty
        max_wait = 3600 * 24  # 24 hours max wait
        start_time = time.time()
        poll_interval = 30  # Check every 30 seconds
        
        while time.time() - start_time < max_wait:
            stats = queue.get_progress()
            pending = stats.get("pending", 0)
            processing = stats.get("processing", 0)
            
            if pending == 0 and processing == 0:
                logger.info("Master: All work items completed")
                break
            
            logger.info(f"Master: Queue status - pending: {pending}, processing: {processing}, completed: {stats.get('completed', 0)}, failed: {stats.get('failed', 0)}")
            time.sleep(poll_interval)
        else:
            logger.warning("Master: Timeout waiting for workers to complete")
        
        # Collect results
        logger.info("Master: Collecting results from completed work items...")
        all_arms: List[Tuple[str, int]] = []
        context_features: Dict[Tuple[str, int], np.ndarray] = {}
        arm_to_model_size: Dict[Tuple[str, int], str] = {}
        feature_keys = ["prompt_entropy", "dataset_entropy", "curvature", "effective_rank"]
        
        # Load all completed metrics
        work_items_list = _generate_all_work_items(cfg, logger)
        for model_size, checkpoint, layer in work_items_list:
            arm = (checkpoint, layer)
            all_arms.append(arm)
            arm_to_model_size[arm] = model_size
            
            # Load metrics from cache
            metrics_path = _cache_metrics_path(run_dir, checkpoint, layer, model_size)
            metrics = _load_metrics(metrics_path)
            
            if metrics is not None:
                feature_vector = np.array([metrics.get(k, 0.0) for k in feature_keys], dtype=np.float32)
                context_features[arm] = feature_vector
            else:
                logger.warning(f"Master: Metrics not found for {checkpoint} layer {layer}, using zero vector")
                context_features[arm] = np.zeros(len(feature_keys), dtype=np.float32)
        
        logger.info(f"Master: Collected metrics for {len(context_features)} arms")
        return context_features, arm_to_model_size
    else:
        # Workers exit after processing
        return {}, {}


def run_mteb_bandit_workflow(cfg: BanditSweepConfig) -> None:
    """
    Run bandit-based MTEB workflow.
    
    Implements the complete workflow:
    1. Build representation corpus
    2. Pre-compute metrics for all (checkpoint, layer) pairs
    3. Run bandit loop to select and evaluate arms
    4. Compute z-scored rewards
    5. Output best arm
    """
    run_dir = cfg.output_root / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "bandit_runner.log"
    logger = setup_logging(log_path)
    
    logger.info(f"Starting bandit workflow | run_dir={run_dir} | budget={cfg.bandit_budget}")
    
    # Set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    # Step 1: Build representation corpus
    logger.info("Step 1: Building representation corpus")
    corpus_cache_path = cfg.corpus_cache_path or (run_dir / "cache" / "representation_corpus.json")
    corpus = load_cached_corpus(corpus_cache_path)
    if corpus is None:
        logger.info(f"Building corpus from {len(cfg.tasks)} tasks")
        corpus = build_representation_corpus(
            cfg.tasks,
            split="train",
            max_examples_per_task=cfg.corpus_max_examples_per_task,
            cache_path=corpus_cache_path,
        )
        logger.info(f"Corpus size: {len(corpus)} examples")
    else:
        logger.info(f"Loaded cached corpus: {len(corpus)} examples")
    
    if len(corpus) == 0:
        raise ValueError("Representation corpus is empty")
    
    # Step 2: Pre-compute metrics for all (checkpoint, layer) pairs
    logger.info("Step 2: Pre-computing metrics for all (checkpoint, layer) pairs")
    
    # Check if we're in distributed mode
    dist_env = _get_distributed_env()
    if dist_env is not None:
        # Distributed mode: use work queue
        logger.info(f"Running in distributed mode: worker {dist_env[0]} of {dist_env[2]}, GPU {dist_env[1]}")
        context_features, arm_to_model_size = _run_distributed_phase1(cfg, corpus, run_dir, logger)
        
        # Only master continues to Phase 2
        if dist_env[0] != 0:
            logger.info(f"Worker {dist_env[0]}: Phase 1 complete, exiting")
            return
        
        # Check if we should exit after Phase 1 (metrics-only mode)
        if os.environ.get("LAYER_TIME_METRICS_ONLY", "false").lower() in ("true", "1", "yes"):
            logger.info("Metrics-only mode: Phase 1 complete, exiting without bandit loop")
            logger.info(f"Computed metrics for {len(context_features)} (checkpoint, layer) pairs")
            return
        
        # Master collects all arms
        all_arms: List[Tuple[str, int]] = []
        work_items_list = _generate_all_work_items(cfg, logger)
        for model_size, checkpoint, layer in work_items_list:
            all_arms.append((checkpoint, layer))
    else:
        # Sequential mode: original code
        logger.info("Running in sequential mode")
        all_arms: List[Tuple[str, int]] = []  # (checkpoint, layer)
        context_features: Dict[Tuple[str, int], np.ndarray] = {}
        arm_to_model_size: Dict[Tuple[str, int], str] = {}  # Map arm to model_size
        
        for model_size in cfg.model_sizes:
            model_id = pythia_model_id(size=model_size, org=cfg.hf_org)
            
            for revision in cfg.revisions:
                logger.info(f"Processing {model_size} @ {revision}")
                
                try:
                    # Load model once per checkpoint
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
                    logger.info(f"Layers: {layers}")
                    
                except Exception as e:
                    # Skip unavailable checkpoints gracefully
                    error_msg = str(e)
                    error_type = type(e).__name__
                    if "RevisionNotFoundError" in error_type or "RevisionNotFoundError" in error_msg or "not a valid git identifier" in error_msg or "404" in error_msg:
                        logger.warning(f"Skipping unavailable checkpoint: {model_size} @ {revision} (not found on HuggingFace)")
                    else:
                        logger.warning(f"Skipping checkpoint {model_size} @ {revision} due to error ({error_type}): {error_msg}")
                    continue
                
                for layer in layers:
                    embedder.set_layer(layer)
                    arm = (revision, layer)
                    all_arms.append(arm)
                    arm_to_model_size[arm] = model_size  # Track which model size this arm belongs to
                    
                    # Compute metrics (using cache)
                    metrics = _compute_metrics_for_checkpoint_layer(
                        embedder,
                        revision,
                        layer,
                        corpus,
                        model_size,
                        run_dir,
                        cfg.batch_size,
                        logger,
                    )
                    
                    # Convert metrics dict to feature vector (use core metrics)
                    feature_keys = ["prompt_entropy", "dataset_entropy", "curvature", "effective_rank"]
                    feature_vector = np.array([metrics.get(k, 0.0) for k in feature_keys], dtype=np.float32)
                    context_features[arm] = feature_vector
                
                # Free GPU memory
                try:
                    import torch
                    del embedder
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        
        logger.info(f"Computed metrics for {len(all_arms)} arms")
    
    logger.info(f"Step 2 complete: {len(all_arms)} arms with metrics")
    
    # Step 3: Initialize bandit
    feature_keys = ["prompt_entropy", "dataset_entropy", "curvature", "effective_rank"]
    context_dim = len(feature_keys)
    bandit = LinUCB(context_dim=context_dim, alpha=cfg.bandit_alpha)
    
    # Load bandit state if resuming
    bandit_state_path = run_dir / "bandit_state.json"
    if cfg.resume and bandit_state_path.exists():
        try:
            bandit = LinUCB.load_state(bandit_state_path)
            logger.info("Loaded bandit state from disk")
        except Exception as e:
            logger.warning(f"Failed to load bandit state: {e}")
    
    # Step 4: Compute baseline scores (for z-scoring)
    logger.info(f"Step 4: Computing baseline scores from {cfg.baseline_checkpoint}")
    
    # Check if baseline exists, if not, we'll compute it on-the-fly
    # For now, assume baseline checkpoint sweep exists or compute it
    baseline_scores: Dict[str, Dict[int, float]] = {}
    baseline_run_dir = run_dir  # Could be different run_dir for baseline
    
    # Try to load baseline
    # Note: Baseline computation requires existing results from final checkpoint sweep
    # For now, we'll compute it on-the-fly or use zero baseline
    baseline_scores: Dict[str, Dict[int, float]] = {}
    try:
        # Try to compute baseline from existing results
        baseline_scores = compute_baseline_scores(
            baseline_run_dir,
            cfg.baseline_checkpoint,
            cfg.model_sizes[0],  # Use first model size
            [],  # Layers will be inferred from results
            cfg.tasks,
        )
        if not baseline_scores:
            logger.warning("Baseline scores empty, will compute on-the-fly or use zero baseline")
    except Exception as e:
        logger.warning(f"Could not load baseline scores: {e}. Will compute on-the-fly or use zero baseline.")
        # Baseline will be computed on-the-fly as we evaluate baseline checkpoint
    
    # Step 5: Bandit loop
    logger.info(f"Step 5: Running bandit loop with budget {cfg.bandit_budget}")
    
    progress_path = run_dir / "bandit_progress.csv"
    if not progress_path.exists():
        pd.DataFrame(columns=[
            "timestamp", "model_size", "arm_checkpoint", "arm_layer", "task", "raw_score",
            "reward", "aggregated_reward", "status"
        ]).to_csv(progress_path, index=False)
    
    evaluated_arms: set = set()
    budget_remaining = cfg.bandit_budget
    
    while budget_remaining > 0:
        # Select arm
        available_arms = [arm for arm in all_arms if arm not in evaluated_arms]
        if not available_arms:
            logger.info("All arms evaluated, stopping")
            break
        
        selected_arm = bandit.select_arm(available_arms, context_features)
        checkpoint, layer = selected_arm
        
        # Get the correct model size for this arm
        model_size = arm_to_model_size.get(selected_arm)
        if model_size is None:
            logger.error(f"Arm {selected_arm} not found in arm_to_model_size mapping. Available arms: {list(arm_to_model_size.keys())[:10]}...")
            evaluated_arms.add(selected_arm)
            budget_remaining -= 1
            continue
        
        logger.info(f"Selected arm: {checkpoint} @ layer {layer} (model: {model_size}, budget remaining: {budget_remaining})")
        
        # Evaluate on all tasks using the correct model size
        model_id = pythia_model_id(size=model_size, org=cfg.hf_org)
        
        # Load embedder with error handling
        try:
            embedder = HFHiddenStateEmbedder(
                model_id=model_id,
                revision=checkpoint,
                pooling=cfg.pooling,
                normalize=cfg.normalize,
                max_length=cfg.max_length,
                batch_size=cfg.batch_size,
                device=cfg.device,
                dtype=cfg.dtype,
                layer_index=layer,
            )
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            if "RevisionNotFoundError" in error_type or "RevisionNotFoundError" in error_msg or "not a valid git identifier" in error_msg or "404" in error_msg:
                logger.error(f"Checkpoint {checkpoint} for {model_size} is no longer available on HuggingFace. Skipping this arm.")
            else:
                logger.error(f"Failed to load model {model_size} @ {checkpoint} (layer {layer}): {error_type}: {error_msg}")
            # Mark as evaluated to skip in future
            evaluated_arms.add(selected_arm)
            budget_remaining -= 1
            continue
        
        task_rewards: Dict[str, float] = {}
        raw_scores: Dict[str, float] = {}
        
        for task_name in cfg.tasks:
            out_dir = (
                run_dir / "outputs" / "mteb" / cfg.model_family
                / model_size / checkpoint / f"layer_{layer:03d}" / task_name
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            done_path = out_dir / "done.json"
            
            # Skip if already done (resume)
            if cfg.resume and done_path.exists():
                logger.debug(f"Skipping {task_name} (already done)")
                # Try to load existing score
                try:
                    from layer_time.analysis.collect_results import _try_load_json, _extract_score_from_result_file
                    result_file = out_dir / "no_model_name_available" / "no_revision_available" / f"{task_name}.json"
                    result_data = _try_load_json(result_file)
                    if result_data:
                        score = _extract_score_from_result_file(result_data)
                        if score is not None:
                            raw_scores[task_name] = score
                except Exception:
                    pass
                continue
            
            try:
                # Run MTEB evaluation
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
                    cache = ResultCache(cache_path=str(out_dir)) if ResultCache else None
                    
                    result = mteb.evaluate(
                        model=embedder,
                        tasks=tasks_obj,
                        cache=cache,
                        overwrite_strategy=OverwriteStrategy.ONLY_MISSING if OverwriteStrategy else "only-missing",
                        encode_kwargs={
                            "batch_size": cfg.batch_size,
                            "max_length": cfg.max_length,
                        },
                        show_progress_bar=False,
                        raise_error=True,
                    )
                    
                    # Save result
                    (out_dir / f"{task_name}.json").write_text(
                        _result_to_pretty_json(result),
                        encoding="utf-8",
                    )
                
                _safe_json_dump({
                    "status": "done",
                    "checkpoint": checkpoint,
                    "layer": layer,
                    "task": task_name,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, done_path)
                
                # Extract score from MTEB result
                # MTEB saves results to: results/EleutherAI__pythia-{size}/{checkpoint}/{task_name}.json
                from layer_time.analysis.collect_results import _try_load_json, _extract_score_from_result_file
                
                # Try the actual MTEB cache location first (preferred)
                model_id_escaped = model_id.replace("/", "__")
                result_file = out_dir / "results" / model_id_escaped / checkpoint / f"{task_name}.json"
                result_data = _try_load_json(result_file)
                
                # If that doesn't work, try the old path format
                if not result_data:
                    result_file = out_dir / "no_model_name_available" / "no_revision_available" / f"{task_name}.json"
                    result_data = _try_load_json(result_file)
                
                # If still not found, try extracting from the saved result object directly
                if not result_data:
                    result_file = out_dir / f"{task_name}.json"
                    saved_result = _try_load_json(result_file)
                    # The saved result might have structure: {"model_name": ..., "task_results": [...]}
                    if saved_result and isinstance(saved_result, dict) and "task_results" in saved_result:
                        task_results = saved_result["task_results"]
                        if isinstance(task_results, list) and len(task_results) > 0:
                            # Take the first task result (should be the one we just ran)
                            result_data = task_results[0]
                
                if result_data:
                    score = _extract_score_from_result_file(result_data)
                    if score is not None:
                        raw_scores[task_name] = score
                        logger.debug(f"Extracted score for {task_name}: {score:.4f}")
                    else:
                        logger.warning(f"Could not extract score from result file structure: {result_file}")
                else:
                    logger.warning(f"Could not find result file for task {task_name} in any expected location")
                
            except Exception as e:
                logger.exception(f"Failed to evaluate {task_name}: {e}")
                if cfg.fail_fast:
                    raise
        
        # Compute rewards
        for task_name, raw_score in raw_scores.items():
            reward = compute_task_reward(
                raw_score,
                task_name,
                baseline_scores,
                layer,
            )
            task_rewards[task_name] = reward
        
        # Aggregate reward
        if task_rewards:
            aggregated_reward = aggregate_rewards(task_rewards, method=cfg.reward_aggregation_method)
            
            # Update bandit
            bandit.update(selected_arm, aggregated_reward, context_features[selected_arm])
            
            # Save progress
            for task_name, raw_score in raw_scores.items():
                reward = task_rewards.get(task_name, 0.0)
                row = pd.DataFrame([{
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_size": model_size,
                    "arm_checkpoint": checkpoint,
                    "arm_layer": layer,
                    "task": task_name,
                    "raw_score": raw_score,
                    "reward": reward,
                    "aggregated_reward": aggregated_reward,
                    "status": "ok",
                }])
                row.to_csv(progress_path, mode="a", header=False, index=False)
            
            logger.info(f"Arm {selected_arm} reward: {aggregated_reward:.4f}")
        else:
            logger.warning(f"No scores for arm {selected_arm}")
        
        evaluated_arms.add(selected_arm)
        budget_remaining -= 1
        
        # Save bandit state periodically
        bandit.save_state(bandit_state_path)
        
        # Free GPU memory
        try:
            import torch
            del embedder
            torch.cuda.empty_cache()
        except Exception:
            pass
    
    # Step 6: Output best arm
    best_arm = bandit.get_best_arm()
    logger.info(f"Best arm: {best_arm}")
    
    # Save final results
    results = {
        "best_arm": {"checkpoint": best_arm[0], "layer": best_arm[1]} if best_arm else None,
        "trajectory_length": len(bandit.get_trajectory()),
        "arms_evaluated": len(evaluated_arms),
        "budget_used": cfg.bandit_budget - budget_remaining,
    }
    _safe_json_dump(results, run_dir / "bandit_results.json")
    
    logger.info("Bandit workflow complete")

