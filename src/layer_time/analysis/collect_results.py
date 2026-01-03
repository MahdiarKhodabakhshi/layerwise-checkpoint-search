"""Aggregate a run directory into a single results table.

This script is intentionally conservative:
- It walks `runs/<run_id>/outputs/mteb/...` for `done.json` markers.
- It optionally parses MTEB output JSON files to extract a main score if possible.

Because MTEB's output schema can vary by version, the score extraction logic is best-effort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _try_load_json(path: Path) -> Optional[Any]:
    """Try to load JSON from a file path, return None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_score_from_result_file(obj: Any) -> Optional[float]:
    """
    Extract main_score from MTEB result file structure.
    
    MTEB result files have structure:
    {
      'scores': {
        'test': [{'main_score': float, ...}, ...],
        'validation': [{'main_score': float, ...}, ...],
        ...
      },
      ...
    }
    
    Prefer test split, fallback to validation, then any split.
    """
    if obj is None or not isinstance(obj, dict):
        return None
    
    # Look for 'scores' key
    if 'scores' not in obj or not isinstance(obj['scores'], dict):
        return None
    
    scores = obj['scores']
    
    # Prefer 'test' split, then 'validation', then any split
    for split_name in ['test', 'validation', 'dev']:
        if split_name in scores and isinstance(scores[split_name], list):
            split_scores = scores[split_name]
            if len(split_scores) > 0 and isinstance(split_scores[0], dict):
                if 'main_score' in split_scores[0]:
                    score = split_scores[0]['main_score']
                    if isinstance(score, (int, float)):
                        return float(score)
    
    # Fallback: try any split
    for split_scores in scores.values():
        if isinstance(split_scores, list) and len(split_scores) > 0:
            if isinstance(split_scores[0], dict) and 'main_score' in split_scores[0]:
                score = split_scores[0]['main_score']
                if isinstance(score, (int, float)):
                    return float(score)
    
        return None


def _extract_score_from_return(obj: Any) -> Optional[float]:
    """
    Best-effort extraction of a numeric main score from MTEB return structures.
    
    This is a fallback for older formats or direct result objects.
    """
    if obj is None:
        return None

    # Try the structured result file format first
    if isinstance(obj, dict):
        score = _extract_score_from_result_file(obj)
        if score is not None:
            return score
        
        # Fallback: search for common score keys
        for k in ["main_score", "score", "cos_sim", "spearman", "accuracy"]:
            if k in obj and isinstance(obj[k], (int, float)):
                return float(obj[k])
        # recurse shallowly
        for v in obj.values():
            s = _extract_score_from_return(v)
            if s is not None:
                return s

    # List/tuple search
    if isinstance(obj, (list, tuple)):
        for v in obj:
            s = _extract_score_from_return(v)
            if s is not None:
                return s

    return None


def collect(run_dir: Path) -> pd.DataFrame:
    outputs_dir = run_dir / "outputs" / "mteb"
    rows = []

    for done_path in outputs_dir.rglob("done.json"):
        done = _try_load_json(done_path) or {}
        out_dir = done_path.parent

        # Try multiple sources for the result file, in order of preference:
        # 1. MTEB cache subdirectory (standard location)
        # 2. Direct {task_name}.json in output directory
        # 3. mteb_return.json (legacy/fallback)
        
        score = None
        task_name = done.get("task", "")
        
        # Parse task name from path if not in done.json
        if not task_name:
            parts = done_path.parts
            try:
                idx = parts.index("mteb")
                task_name = parts[idx + 5]
            except Exception:
                pass
        
        # Try MTEB cache subdirectory first (standard location)
        if task_name:
            cache_result_file = out_dir / "no_model_name_available" / "no_revision_available" / f"{task_name}.json"
            result_data = _try_load_json(cache_result_file)
            if result_data:
                score = _extract_score_from_result_file(result_data)
        
        # Try direct {task_name}.json in output directory
        if score is None and task_name:
            direct_result_file = out_dir / f"{task_name}.json"
            result_data = _try_load_json(direct_result_file)
            if result_data:
                score = _extract_score_from_result_file(result_data)
        
        # Fallback to mteb_return.json (may have different structure)
        if score is None:
            mteb_return = _try_load_json(out_dir / "mteb_return.json")
            score = _extract_score_from_return(mteb_return)

        # Parse path components: .../Pythia/410m/main/layer_003/TaskName/done.json
        parts = done_path.parts
        # Find indices robustly
        try:
            idx = parts.index("mteb")
            model_family = parts[idx + 1]
            model_size = parts[idx + 2]
            revision = parts[idx + 3]
            layer_str = parts[idx + 4]  # layer_XXX
            task = parts[idx + 5]
            layer = int(layer_str.split("_")[1])
        except Exception:
            model_family = done.get("model_family", "")
            model_size = done.get("model_size", "")
            revision = done.get("revision", "")
            layer = int(done.get("layer", -1))
            task = done.get("task", "")

        rows.append(
            {
                "model_family": model_family,
                "model_size": model_size,
                "revision": revision,
                "layer": layer,
                "task": task,
                "score": score,
                "output_dir": str(out_dir),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str, required=True, help="Path to runs/<run_id>")
    ap.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    df = collect(run_dir)
    df.to_csv(Path(args.out), index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
