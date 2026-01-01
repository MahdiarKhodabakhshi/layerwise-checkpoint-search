"""Constants used across experiments.

The task list below matches **Table 1** in:
"Layer by Layer: Uncovering Hidden Representations in Language Models" (Skean et al., 2025).

Keep this file as the single source of truth for the benchmark task set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# ---------------------------------------------------------------------
# MTEB task presets
# ---------------------------------------------------------------------

# Exactly the 32 tasks listed in Table 1 of the paper.
# Grouped for readability; the runner flattens the list.
LAYER_BY_LAYER_MTEB_32: Dict[str, List[str]] = {
    "pair_classification": [
        "SprintDuplicateQuestions",
        "TwitterSemEval2015",
        "TwitterURLCorpus",
    ],
    "classification": [
        "AmazonCounterfactualClassification",
        "AmazonReviewsClassification",
        "Banking77Classification",
        "EmotionClassification",
        "MTOPDomainClassification",
        "MTOPIntentClassification",
        "MassiveIntentClassification",
        "MassiveScenarioClassification",
        "ToxicConversationsClassification",
        "TweetSentimentExtractionClassification",
    ],
    "clustering": [
        "ArxivClusteringS2S",
        "BiorxivClusteringS2S",
        "MedrxivClusteringS2S",
        "RedditClustering",
        "StackExchangeClustering",
        "TwentyNewsgroupsClustering",
    ],
    "reranking": [
        "AskUbuntuDupQuestions",
        "MindSmallReranking",
        "SciDocsRR",
        "StackOverflowDupQuestions",
    ],
    "sts": [
        "BIOSSES",
        "SICK-R",
        "STS12",
        "STS13",
        "STS14",
        "STS15",
        "STS16",
        "STS17",
        "STSBenchmark",
    ],
}


def flatten_task_preset(preset: Dict[str, List[str]]) -> List[str]:
    """Flatten a grouped task preset into an ordered list."""
    tasks: List[str] = []
    for _, group in preset.items():
        tasks.extend(group)
    return tasks


TASK_PRESETS: Dict[str, List[str]] = {
    "layer_by_layer_32": flatten_task_preset(LAYER_BY_LAYER_MTEB_32),
}


# ---------------------------------------------------------------------
# HuggingFace model id helpers
# ---------------------------------------------------------------------

def pythia_model_id(size: str, org: str = "EleutherAI") -> str:
    """Return the HuggingFace model id for a Pythia size string.

    Examples:
      size="14m"  -> "EleutherAI/pythia-14m"
      size="410m" -> "EleutherAI/pythia-410m"
    """
    size = size.strip().lower()
    return f"{org}/pythia-{size}"
