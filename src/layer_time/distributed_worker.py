"""
Distributed worker implementation for parallel metric computation.

This module enables parallel processing of (checkpoint, layer) pairs across
multiple GPUs/nodes using a file-based work queue on shared filesystem.
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """A single work item: compute metrics for (checkpoint, layer) pair."""
    model_size: str
    checkpoint: str
    layer: int
    worker_id: Optional[int] = None
    status: str = "pending"  # pending, processing, completed, failed
    result_path: Optional[str] = None


class WorkQueue:
    """File-based work queue for distributed processing."""
    
    def __init__(self, queue_dir: Path, num_workers: int):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        
        self.pending_file = self.queue_dir / "pending.jsonl"
        self.processing_file = self.queue_dir / "processing.jsonl"
        self.completed_file = self.queue_dir / "completed.jsonl"
        self.failed_file = self.queue_dir / "failed.jsonl"
        self.lock_file = self.queue_dir / ".lock"
    
    def _acquire_lock(self, timeout: float = 30.0) -> bool:
        """Acquire file lock (simple timeout-based)."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                if not self.lock_file.exists():
                    self.lock_file.touch()
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False
    
    def _release_lock(self):
        """Release file lock."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception:
            pass
    
    def enqueue(self, work_items: List[WorkItem]):
        """Add work items to the queue."""
        if not self._acquire_lock():
            raise RuntimeError("Failed to acquire lock for enqueue")
        
        try:
            # Append to pending file
            with open(self.pending_file, "a") as f:
                for item in work_items:
                    f.write(json.dumps({
                        "model_size": item.model_size,
                        "checkpoint": item.checkpoint,
                        "layer": item.layer,
                        "status": item.status,
                    }) + "\n")
        finally:
            self._release_lock()
    
    def dequeue(self, worker_id: int) -> Optional[WorkItem]:
        """Get next work item from queue (atomic operation)."""
        if not self._acquire_lock():
            return None
        
        try:
            # Read pending items
            pending_items = []
            if self.pending_file.exists():
                with open(self.pending_file, "r") as f:
                    for line in f:
                        if line.strip():
                            pending_items.append(json.loads(line))
            
            if not pending_items:
                return None
            
            # Take first item
            item_data = pending_items[0]
            remaining = pending_items[1:]
            
            # Write remaining back
            with open(self.pending_file, "w") as f:
                for item in remaining:
                    f.write(json.dumps(item) + "\n")
            
            # Move to processing
            with open(self.processing_file, "a") as f:
                item_data["worker_id"] = worker_id
                item_data["status"] = "processing"
                f.write(json.dumps(item_data) + "\n")
            
            return WorkItem(
                model_size=item_data["model_size"],
                checkpoint=item_data["checkpoint"],
                layer=item_data["layer"],
                worker_id=worker_id,
                status="processing",
            )
        finally:
            self._release_lock()
    
    def mark_completed(self, item: WorkItem, result_path: str):
        """Mark work item as completed."""
        if not self._acquire_lock():
            return
        
        try:
            # Remove from processing
            processing_items = []
            if self.processing_file.exists():
                with open(self.processing_file, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if not (data["model_size"] == item.model_size and
                                   data["checkpoint"] == item.checkpoint and
                                   data["layer"] == item.layer):
                                processing_items.append(data)
            
            with open(self.processing_file, "w") as f:
                for data in processing_items:
                    f.write(json.dumps(data) + "\n")
            
            # Add to completed
            with open(self.completed_file, "a") as f:
                f.write(json.dumps({
                    "model_size": item.model_size,
                    "checkpoint": item.checkpoint,
                    "layer": item.layer,
                    "worker_id": item.worker_id,
                    "status": "completed",
                    "result_path": result_path,
                }) + "\n")
        finally:
            self._release_lock()
    
    def mark_failed(self, item: WorkItem, error: str):
        """Mark work item as failed."""
        if not self._acquire_lock():
            return
        
        try:
            # Remove from processing
            processing_items = []
            if self.processing_file.exists():
                with open(self.processing_file, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if not (data["model_size"] == item.model_size and
                                   data["checkpoint"] == item.checkpoint and
                                   data["layer"] == item.layer):
                                processing_items.append(data)
            
            with open(self.processing_file, "w") as f:
                for data in processing_items:
                    f.write(json.dumps(data) + "\n")
            
            # Add to failed
            with open(self.failed_file, "a") as f:
                f.write(json.dumps({
                    "model_size": item.model_size,
                    "checkpoint": item.checkpoint,
                    "layer": item.layer,
                    "worker_id": item.worker_id,
                    "status": "failed",
                    "error": error,
                }) + "\n")
        finally:
            self._release_lock()
    
    def get_progress(self) -> Dict[str, int]:
        """Get progress statistics."""
        stats = {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
        }
        
        for file, key in [
            (self.pending_file, "pending"),
            (self.processing_file, "processing"),
            (self.completed_file, "completed"),
            (self.failed_file, "failed"),
        ]:
            if file.exists():
                with open(file, "r") as f:
                    stats[key] = sum(1 for line in f if line.strip())
        
        return stats


def run_distributed_worker(
    queue_dir: Path,
    worker_id: int,
    gpu_id: int,
    corpus: List[str],
    run_dir: Path,
    batch_size: int,
    cfg: Any,
    logger: logging.Logger,
) -> None:
    """Run a distributed worker that processes work items from the queue."""
    
    queue = WorkQueue(queue_dir, num_workers=1)  # Not used for num_workers
    
    # Set GPU device
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    logger.info(f"Worker {worker_id} using GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
    
    from layer_time.embedder import HFHiddenStateEmbedder
    from layer_time.metrics import compute_representation_metrics
    from layer_time.constants import pythia_model_id
    from layer_time.mteb_bandit_runner import (
        _cache_embeddings_path,
        _cache_metrics_path,
        _save_embeddings,
        _save_metrics,
        _load_embeddings,
        _extract_embeddings_for_corpus,
    )
    
    processed = 0
    consecutive_empty = 0
    max_empty = 10  # Stop after 10 consecutive empty polls
    
    while consecutive_empty < max_empty:
        item = queue.dequeue(worker_id)
        
        if item is None:
            consecutive_empty += 1
            time.sleep(1.0)
            continue
        
        consecutive_empty = 0
        logger.info(f"Worker {worker_id}: Processing {item.model_size} @ {item.checkpoint} layer {item.layer}")
        
        try:
            model_id = pythia_model_id(size=item.model_size, org=cfg.hf_org)
            
            embedder = HFHiddenStateEmbedder(
                model_id=model_id,
                revision=item.checkpoint,
                pooling=cfg.pooling,
                normalize=cfg.normalize,
                max_length=cfg.max_length,
                batch_size=batch_size,
                device=device,
                dtype=cfg.dtype,
                layer_index=item.layer,
            )
            
            metrics_path = _cache_metrics_path(run_dir, item.checkpoint, item.layer, item.model_size)
            if metrics_path.exists():
                logger.info(f"Worker {worker_id}: Metrics already cached for {item.checkpoint} layer {item.layer}")
                queue.mark_completed(item, str(metrics_path))
                del embedder
                torch.cuda.empty_cache()
                processed += 1
                continue
            
            embeddings_path = _cache_embeddings_path(run_dir, item.checkpoint, item.layer, item.model_size)
            embeddings = _load_embeddings(embeddings_path)
            
            if embeddings is None:
                logger.info(f"Worker {worker_id}: Extracting embeddings for {item.checkpoint} layer {item.layer}")
                embeddings = _extract_embeddings_for_corpus(embedder, corpus, batch_size)
                _save_embeddings(embeddings, embeddings_path)
            
            logger.info(f"Worker {worker_id}: Computing metrics for {item.checkpoint} layer {item.layer}")
            metrics = compute_representation_metrics(embeddings)
            _save_metrics(metrics, metrics_path)
            
            queue.mark_completed(item, str(metrics_path))
            processed += 1
            
            del embedder
            torch.cuda.empty_cache()
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Worker {worker_id}: Failed to process {item.checkpoint} layer {item.layer}: {error_msg}")
            queue.mark_failed(item, error_msg)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        
        if processed % 10 == 0:
            stats = queue.get_progress()
            logger.info(f"Worker {worker_id}: Processed {processed} items. Queue: {stats}")
    
    logger.info(f"Worker {worker_id}: Finished. Processed {processed} items total.")
    
    stats = queue.get_progress()
    logger.info(f"Worker {worker_id}: Final queue stats: {stats}")