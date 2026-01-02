"""Representation corpus selection and management.

Step 2 of the workflow: Pick a representation corpus D_repr (cheap, reused everywhere).
For each downstream task, use the task's train split texts (or aggregate across tasks)
as the fixed corpus from which representation metrics are computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import mteb
from mteb.abstasks import AbsTask


def build_representation_corpus(
    tasks: List[str],
    split: str = "train",
    max_examples_per_task: Optional[int] = None,
    cache_path: Optional[Path] = None,
) -> List[str]:
    """
    Build a fixed representation corpus from task train splits.
    
    This corpus is reused for ALL (checkpoint, layer) pairs to compute representation
    metrics. This ensures metrics are comparable across different checkpoints/layers.
    
    Args:
        tasks: List of MTEB task names
        split: Data split to use (default: "train")
        max_examples_per_task: Optional limit on examples per task (for memory efficiency)
        cache_path: Optional path to cache the corpus text
    
    Returns:
        List of text strings (the representation corpus)
    
    Example:
        >>> tasks = ["STS12", "STS13", "Banking77Classification"]
        >>> corpus = build_representation_corpus(tasks, split="train", max_examples_per_task=1000)
        >>> print(f"Corpus size: {len(corpus)}")
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and all(isinstance(x, str) for x in cached):
                return cached
        except Exception:
            pass  # Cache invalid, rebuild
    
    all_texts: List[str] = []
    
    for task_name in tasks:
        try:
            task_objs = mteb.get_tasks(tasks=[task_name])
            if not task_objs:
                continue
            
            task = task_objs[0]  # Get first (should be only one)
            
            # Load the dataset (load_data returns None, loads into task.dataset)
            try:
                task.load_data()
            except Exception as e:
                print(f"Warning: Failed to load data for task {task_name}: {e}")
                continue
            
            if not hasattr(task, 'dataset') or task.dataset is None:
                continue
            
            dataset = task.dataset
            
            # Handle different dataset structures:
            # - Monolingual: {"split": Dataset}
            # - Multilingual: {"hf_subset": {"split": Dataset}}
            
            split_data = None
            if isinstance(dataset, dict):
                # Try to find the requested split
                if split in dataset:
                    split_data = dataset[split]
                else:
                    # Try multilingual structure
                    for hf_subset in dataset.values():
                        if isinstance(hf_subset, dict) and split in hf_subset:
                            split_data = hf_subset[split]
                            break
                    # If still not found, try "test" split as fallback
                    if split_data is None:
                        if "test" in dataset:
                            split_data = dataset["test"]
                        else:
                            for hf_subset in dataset.values():
                                if isinstance(hf_subset, dict) and "test" in hf_subset:
                                    split_data = hf_subset["test"]
                                    break
            
            if split_data is None:
                continue
            
            # Extract texts from Dataset object
            texts: List[str] = []
            
            # Handle HuggingFace Dataset objects
            from datasets import Dataset as HFDataset
            if isinstance(split_data, HFDataset):
                # Get column names
                columns = split_data.column_names
                
                # Try common text column names
                text_columns = ["text", "sentence", "sentence1", "sentence2", "sentences", "input", "query"]
                
                for col in text_columns:
                    if col in columns:
                        texts.extend(split_data[col])
                        break
                
                # If no single text column, try combining sentence1 + sentence2 (for STS tasks)
                if not texts and "sentence1" in columns and "sentence2" in columns:
                    texts.extend(split_data["sentence1"])
                    texts.extend(split_data["sentence2"])
                
                # Try title + text combination
                if not texts and "title" in columns and "text" in columns:
                    titles = split_data["title"]
                    texts_list = split_data["text"]
                    texts = [f"{t} {x}".strip() if t else x for t, x in zip(titles, texts_list)]
            
            # Handle list of dicts
            elif isinstance(split_data, list):
                for ex in split_data:
                    if isinstance(ex, dict):
                        # Try common text fields
                        for key in ["text", "sentence", "sentence1", "sentence2", "sentences", "input"]:
                            if key in ex:
                                val = ex[key]
                                if isinstance(val, str):
                                    texts.append(val)
                                elif isinstance(val, list):
                                    texts.extend([v for v in val if isinstance(v, str)])
                        # Handle title + text
                        if "title" in ex and "text" in ex:
                            title = ex["title"] if isinstance(ex["title"], str) else ""
                            text = ex["text"] if isinstance(ex["text"], str) else ""
                            texts.append(f"{title} {text}".strip())
                    elif isinstance(ex, str):
                        texts.append(ex)
            
            # Remove empty strings
            texts = [t for t in texts if t and isinstance(t, str) and t.strip()]
            
            # Apply per-task limit if specified
            if max_examples_per_task is not None and len(texts) > max_examples_per_task:
                texts = texts[:max_examples_per_task]
            
            all_texts.extend(texts)
            
        except Exception as e:
            # Log but continue with other tasks
            print(f"Warning: Failed to load corpus from task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Cache if path provided
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(all_texts, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return all_texts


def load_cached_corpus(cache_path: Path) -> Optional[List[str]]:
    """Load a cached representation corpus."""
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

