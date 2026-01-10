# Models, Metrics, and Tasks Overview

## 🤖 Models Being Tested

### Model Family: Pythia (EleutherAI)

The project evaluates **Pythia models** from EleutherAI, specifically testing different checkpoint-layer combinations.

### Model Sizes (Configurable)

From `configs/mteb_layersweep.yaml`, default model sizes:
- **14m** (14 million parameters) - Smallest, fastest
- **70m** (70 million parameters) - Medium
- **410m** (410 million parameters) - Larger, more capable

Model IDs follow the pattern: `EleutherAI/pythia-{size}`

### Checkpoints/Revisions

- **Default**: `"main"` (final trained checkpoint)
- **Expandable**: Can add intermediate training checkpoints if available on HuggingFace

Example configurations:
```yaml
hf:
  model_sizes: ["14m", "70m", "410m"]
  revisions: ["main"]  # Can add: ["step-1000", "step-2000", ..., "main"]
```

### What's Being Evaluated

For each model, the project evaluates:
- **All transformer layers** (layer 0 to num_hidden_layers-1)
- **Different checkpoints** (if multiple revisions specified)
- **Each (checkpoint, layer) pair** is treated as a candidate embedding model

**Search Space**: `A = {(checkpoint, layer): checkpoint ∈ T, layer ∈ [0, L-1]}`

---

## 📊 Representation Metrics

The project computes **stationary (label-free) representation metrics** for each (checkpoint, layer) pair. These metrics serve as features for the bandit algorithm.

### Core Metrics (Required)

1. **Prompt Entropy**
   - Measures the diversity/spread of representations
   - Computed using entropy of the embedding distribution
   - Higher entropy = more diverse representations

2. **Dataset Entropy**
   - Measures entropy of the entire dataset's embedding distribution
   - Similar to prompt entropy but computed differently
   - Uses pairwise distance distribution

3. **Curvature**
   - Measures the geometric curvature of the embedding manifold
   - Computed using nearest neighbors and covariance
   - Higher curvature = more complex geometry

4. **Effective Rank**
   - Measures the dimensionality of the representation space
   - Uses singular value decomposition
   - Effective rank = exp(entropy of normalized singular values)
   - Higher effective rank = higher dimensional representation

### Optional Metrics

5. **Log-Determinant of Covariance**
   - Measures volume of the embedding space
   - Computed from covariance matrix

6. **Anisotropy**
   - Measures variance of singular values
   - Higher anisotropy = more directional representations

7. **Spectral Norm**
   - Maximum singular value of covariance matrix
   - Measures the dominant direction strength

8. **Mean Pairwise Cosine Similarity**
   - Average cosine similarity between embedding pairs
   - Measures how similar/dissimilar embeddings are

### Metric Computation

- Metrics are computed on a **fixed representation corpus** (Step 2 of workflow)
- Corpus is built from task train splits (aggregated across all tasks)
- Metrics are cached per (checkpoint, layer) for efficiency
- Core metrics (4) are used as context features for the bandit algorithm

**Implementation**: `src/layer_time/metrics.py`

---

## 📝 Tasks (MTEB Benchmark)

The project evaluates on **32 MTEB tasks** from the "Layer by Layer" paper (Table 1).

### Task Categories

#### 1. Pair Classification (3 tasks)
- `SprintDuplicateQuestions`
- `TwitterSemEval2015`
- `TwitterURLCorpus`

#### 2. Classification (10 tasks)
- `AmazonCounterfactualClassification`
- `AmazonReviewsClassification`
- `Banking77Classification`
- `EmotionClassification`
- `MTOPDomainClassification`
- `MTOPIntentClassification`
- `MassiveIntentClassification`
- `MassiveScenarioClassification`
- `ToxicConversationsClassification`
- `TweetSentimentExtractionClassification`

#### 3. Clustering (6 tasks)
- `ArxivClusteringS2S`
- `BiorxivClusteringS2S`
- `MedrxivClusteringS2S`
- `RedditClustering`
- `StackExchangeClustering`
- `TwentyNewsgroupsClustering`

#### 4. Reranking (4 tasks)
- `AskUbuntuDupQuestions`
- `MindSmallReranking`
- `SciDocsRR`
- `StackOverflowDupQuestions`

#### 5. Semantic Textual Similarity (STS) (9 tasks)
- `BIOSSES`
- `SICK-R`
- `STS12`
- `STS13`
- `STS14`
- `STS15`
- `STS16`
- `STS17`
- `STSBenchmark`

### Task Details

- **Total**: 32 tasks
- **Source**: "Layer by Layer: Uncovering Hidden Representations in Language Models" paper (Table 1)
- **Preset Name**: `"layer_by_layer_32"` in config
- **Evaluation**: Each task is evaluated using MTEB benchmark framework
- **Metrics per Task**: Each task has its own evaluation metric (accuracy, F1, cosine similarity, etc.)

**Task List**: Defined in `src/layer_time/constants.py`

---

## 🔄 Evaluation Workflow

### Bandit Workflow Mode

1. **Pre-compute metrics** for all (checkpoint, layer) pairs
2. **Bandit algorithm** selects which (checkpoint, layer) to evaluate
3. **Evaluate** selected pairs on all 32 MTEB tasks
4. **Compute rewards** (z-scored relative to baseline)
5. **Output** best (checkpoint, layer) pair

### Brute-Force Mode

1. **Evaluate** all (checkpoint, layer, task) combinations
2. **Store results** for each combination
3. **No intelligent selection** - exhaustive evaluation

---

## 📈 Output Metrics

### Per-Task Metrics

Each task evaluation produces:
- **Main Score**: Task-specific metric (accuracy, F1, cosine similarity, etc.)
- Extracted from MTEB results as `main_score` from `scores.test[0].main_score` or `scores.validation[0].main_score`

### Aggregated Metrics

For bandit workflow:
- **Z-scored Rewards**: Normalized relative to final checkpoint baseline
- **Aggregated Reward**: Mean (or harmonic/robust mean) across tasks
- **Best Arm**: (checkpoint, layer) with highest aggregated reward

For brute-force mode:
- **Raw Scores**: Per (checkpoint, layer, task) combination
- **Aggregated**: Can be computed post-hoc from results

---

## 🎯 Summary

| Component | Details |
|-----------|---------|
| **Models** | Pythia (14m, 70m, 410m) from EleutherAI |
| **Checkpoints** | "main" (final) + optionally intermediate checkpoints |
| **Layers** | All transformer layers (0 to num_hidden_layers-1) |
| **Representation Metrics** | 4 core (prompt entropy, dataset entropy, curvature, effective rank) + 4 optional |
| **Tasks** | 32 MTEB tasks across 5 categories (pair classification, classification, clustering, reranking, STS) |
| **Evaluation Metrics** | Task-specific (accuracy, F1, cosine similarity, etc.) |
| **Rewards** | Z-scored relative to baseline, aggregated across tasks |

---

## 📚 References

- **Paper**: "Layer by Layer: Uncovering Hidden Representations in Language Models" (Skean et al., 2025)
- **Repository**: https://github.com/OFSkean/information_flow
- **MTEB Benchmark**: https://github.com/embeddings-benchmark/mteb
- **Pythia Models**: https://huggingface.co/EleutherAI/pythia-14m (and variants)


