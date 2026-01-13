from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# MTEB expects models to be instances of its Encoder base class.
# Import path differs a bit across MTEB versions, so we try a few.
try:
    from mteb.models import Encoder as MTEBEncoder  # type: ignore
except Exception:
    try:
        from mteb.models.encoder import Encoder as MTEBEncoder  # type: ignore
    except Exception:
        MTEBEncoder = object  # fallback (should not happen on your setup)

# Model metadata (path differs across MTEB versions)
try:
    from mteb.models.model_meta import ModelMeta  # type: ignore
except Exception:
    from mteb.models import ModelMeta  # type: ignore


TextLike = Union[str, Mapping[str, Any]]


def _is_dataloader(x: Any) -> bool:
    return hasattr(x, "__iter__") and hasattr(x, "__len__") and hasattr(x, "batch_size")


def _convert_column_to_python(obj: Any) -> Any:
    """
    Recursively convert datasets.Column objects to Python lists/dicts.
    This is needed because MTEB v1.14.19 passes Column objects directly.
    """
    try:
        from datasets.arrow_dataset import Column
        if isinstance(obj, Column):
            # Convert Column to list and recursively process elements
            return [_convert_column_to_python(x) for x in obj]
        elif isinstance(obj, (list, tuple)):
            return [_convert_column_to_python(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _convert_column_to_python(v) for k, v in obj.items()}
        else:
            return obj
    except ImportError:
        return obj


def _normalize_inputs_to_texts(items: Any) -> List[str]:
    """
    Convert common MTEB batch formats to list[str].
    Handles:
      - list[str]
      - list[dict] with keys like text/title/sentence/sentences
      - dict batch from DataLoader with fields containing lists
      - tuple/list batches (take first element if it looks like texts)
      - HuggingFace datasets.Column objects (MTEB v1.14.19)
    """
    # Handle HuggingFace datasets.Column objects (MTEB v1.14.19 passes these)
    # Convert at the top level first - this handles all nested Column objects
    try:
        from datasets.arrow_dataset import Column
        if isinstance(items, Column):
            # Column is iterable, convert to list and recurse
            items = _convert_column_to_python(items)
            # After conversion, items should be a list, so continue processing
        elif isinstance(items, (list, tuple, dict)):
            # Check if any nested element is a Column and convert
            items = _convert_column_to_python(items)
    except ImportError:
        pass  # datasets not available, skip this check
    
    if isinstance(items, str):
        return [items]

    if isinstance(items, Mapping):
        # Convert any Column objects in the mapping (already done by _convert_column_to_python above)
        # But handle case where items wasn't converted yet
        try:
            from datasets.arrow_dataset import Column
            # Double-check for any remaining Column objects
            items = {k: (_convert_column_to_python(v) if isinstance(v, Column) else v) for k, v in items.items()}
        except ImportError:
            pass
        
        for key in ("sentences", "sentence", "text"):
            if key in items:
                val = items[key]
                if isinstance(val, str):
                    return [val]
                if isinstance(val, (list, tuple)):
                    if len(val) == 0:
                        return []
                    if isinstance(val[0], str):
                        return list(val)

        if "title" in items and "text" in items:
            titles = items["title"]
            texts = items["text"]
            # Handle datasets.Column objects - convert to lists
            try:
                from datasets.arrow_dataset import Column
                if isinstance(titles, Column):
                    titles = list(titles)
                if isinstance(texts, Column):
                    texts = list(texts)
            except ImportError:
                pass
            if isinstance(titles, str) and isinstance(texts, str):
                return [f"{titles} {texts}".strip()]
            if isinstance(titles, (list, tuple)) and isinstance(texts, (list, tuple)):
                out: List[str] = []
                for t, x in zip(titles, texts):
                    # Handle nested Column objects in the lists
                    try:
                        from datasets.arrow_dataset import Column
                        if isinstance(t, Column):
                            t = list(t)[0] if len(list(t)) > 0 else str(t)
                        if isinstance(x, Column):
                            x = list(x)[0] if len(list(x)) > 0 else str(x)
                    except ImportError:
                        pass
                    out.append(f"{t} {x}".strip())
                return out

        if "input" in items and isinstance(items["input"], str):
            return [items["input"]]
        
        # Reranking tasks: handle query-document pairs
        if "query" in items and "document" in items:
            query = items["query"]
            doc = items["document"]
            # Handle Column objects
            try:
                from datasets.arrow_dataset import Column
                if isinstance(query, Column):
                    query = list(query)
                if isinstance(doc, Column):
                    doc = list(doc)
            except ImportError:
                pass
            if isinstance(query, str) and isinstance(doc, str):
                return [f"{query} {doc}".strip()]
            if isinstance(query, (list, tuple)) and isinstance(doc, (list, tuple)):
                out: List[str] = []
                for q, d in zip(query, doc):
                    out.append(f"{q} {d}".strip())
                return out
        
        if "query" in items:
            # Just query (for reranking, document might be passed separately)
            query = items["query"]
            try:
                from datasets.arrow_dataset import Column
                if isinstance(query, Column):
                    query = list(query)
            except ImportError:
                pass
            if isinstance(query, str):
                return [query]
            if isinstance(query, (list, tuple)):
                return [str(q) for q in query]

        raise ValueError(f"Don't know how to extract texts from mapping keys={list(items.keys())[:10]}")

    if isinstance(items, (list, tuple)):
        if len(items) == 0:
            return []

        # Handle list/tuple of Column objects (convert first, then process)
        try:
            from datasets.arrow_dataset import Column
            # Check if any element is a Column and convert all
            has_columns = any(isinstance(x, Column) for x in items)
            if has_columns:
                # Convert all Column objects to lists
                items = [list(col) if isinstance(col, Column) else col for col in items]
                # Recursively process the converted items
                return _normalize_inputs_to_texts(items)
        except ImportError:
            pass

        if isinstance(items[0], str):
            return list(items)

        if isinstance(items[0], Mapping):
            out: List[str] = []
            for ex in items:
                # Handle Column objects in dict elements - convert to dict first
                try:
                    from datasets.arrow_dataset import Column
                    if isinstance(ex, Column):
                        # Column might be a dict-like structure, try to convert
                        ex_list = list(ex)
                        if len(ex_list) > 0 and isinstance(ex_list[0], dict):
                            ex = ex_list[0]
                        elif len(ex_list) > 0:
                            # Single item, wrap it
                            ex = {"text": str(ex_list[0])}
                        else:
                            continue
                except (ImportError, TypeError, IndexError):
                    pass
                
                if not isinstance(ex, Mapping):
                    continue
                    
                if "text" in ex:
                    out.append(str(ex["text"]))
                elif "sentence" in ex:
                    out.append(str(ex["sentence"]))
                elif "sentences" in ex and isinstance(ex["sentences"], str):
                    out.append(ex["sentences"])
                elif "title" in ex and "text" in ex:
                    out.append(f"{ex['title']} {ex['text']}".strip())
                elif "query" in ex and "document" in ex:
                    # Reranking tasks: combine query and document
                    out.append(f"{ex['query']} {ex['document']}".strip())
                elif "query" in ex and "positive" in ex:
                    # Reranking tasks: query with positive document
                    pos = ex["positive"]
                    if isinstance(pos, (list, tuple)) and len(pos) > 0:
                        pos = pos[0]
                    out.append(f"{ex['query']} {pos}".strip())
                elif "query" in ex:
                    # Reranking tasks: just query (document might be separate)
                    out.append(str(ex["query"]))
                else:
                    raise ValueError(f"Unknown dict example keys={list(ex.keys())}")
            return out

        if isinstance(items[0], (list, tuple)) and len(items[0]) > 0:
            # Check if it's a list of tuples (e.g., reranking: [(query, doc), ...])
            if isinstance(items[0][0], (list, tuple)) and len(items[0][0]) == 2:
                # Likely reranking format: [(query, doc), ...]
                out: List[str] = []
                for pair in items:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        # Combine query and document
                        q, d = pair[0], pair[1]
                        out.append(f"{q} {d}".strip())
                    else:
                        # Fallback: just use the first element
                        out.append(str(pair[0]) if len(pair) > 0 else "")
                return out
            elif isinstance(items[0][0], str):
                return list(items[0])

    raise ValueError(f"Don't know how to normalize inputs of type {type(items)}")


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean pooling over LAST tokens (matching paper's implementation).
    
    The paper pools over the last 'length' tokens from the END of the sequence,
    not all non-padding tokens. For causal models, last tokens have more context.
    
    Paper's implementation: hidden_states[i, -length:, :].mean(dim=0)
    """
    seq_lengths = attention_mask.sum(dim=-1)  # [B] - actual sequence lengths
    return torch.stack(
        [
            last_hidden[i, -length:, :].mean(dim=0)  # Pool over LAST 'length' tokens
            for i, length in enumerate(seq_lengths)
        ],
        dim=0,
    )


def _infer_num_transformer_layers(cfg: Any) -> int:
    # Covers common HF config field names across architectures
    for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    raise AttributeError(
        f"Cannot infer number of transformer layers from config fields. "
        f"Available keys: {list(getattr(cfg, '__dict__', {}).keys())[:50]}"
    )


@dataclass
class HFHiddenStateEmbedder(MTEBEncoder):
    model_id: str
    revision: str = "main"
    pooling: str = "mean"  # "mean" or "cls"
    normalize: bool = True
    max_length: int = 2048  # Match paper's max_sample_length for causal models
    batch_size: int = 64
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"

    # IMPORTANT: this code maps layer_index -> hidden_states[layer_index + 1]
    # so layer_index is over transformer blocks only (excluding hidden_states[0] embeddings).
    layer_index: int = 0  # 0..num_hidden_layers-1

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._n_transformer_layers = None  # cached int
        self._resolved_device = None  # cached str

    @property
    def model_name(self) -> str:
        return f"{self.model_id}@{self.revision}-layer{self.layer_index}"

    def _device_str(self) -> str:
        """
        Resolve device strings. Accepts 'auto' and picks CUDA if available.
        """
        if getattr(self, "_resolved_device", None) is not None:
            return str(self._resolved_device)

        d = str(getattr(self, "device", "auto")).strip().lower()
        if d in ("", "auto", "none"):
            d = "cuda" if torch.cuda.is_available() else "cpu"

        self._resolved_device = d
        return d

    @property
    def num_hidden_layers(self) -> int:
        """
        Number of transformer blocks (does NOT count embeddings hidden_states[0]).

        This matches your indexing:
          idx = layer_index + 1
        so valid layer_index is [0 .. num_hidden_layers-1].
        """
        if getattr(self, "_n_transformer_layers", None) is None:
            if self._model is not None:
                self._n_transformer_layers = _infer_num_transformer_layers(self._model.config)
            else:
                cfg = AutoConfig.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    trust_remote_code=False,
                )
                self._n_transformer_layers = _infer_num_transformer_layers(cfg)
        return int(self._n_transformer_layers)

    def set_layer(self, layer_index: int) -> None:
        n = self.num_hidden_layers
        layer_index = int(layer_index)
        if layer_index < 0 or layer_index >= n:
            raise ValueError(f"layer_index={layer_index} out of range [0, {n-1}]")
        self.layer_index = layer_index

    @property
    def mteb_model_meta(self) -> ModelMeta:
        model = getattr(self, "_model", None)
        cfg = getattr(model, "config", None) if model is not None else None
        embed_dim = getattr(cfg, "hidden_size", None) if cfg is not None else None

        return ModelMeta(
            loader=None,
            name=self.model_id,
            revision=self.revision,
            release_date=None,
            languages=None,
            n_parameters=None,
            memory_usage_mb=None,
            max_tokens=self.max_length,
            embed_dim=embed_dim,
            license=None,
            open_weights=True,
            public_training_code=None,
            public_training_data=None,
            framework=["PyTorch"],
            reference=f"https://huggingface.co/{self.model_id}",
            similarity_fn_name="cosine",
            use_instructions=False,
            training_datasets=None,
            modalities=["text"],
        )

    def similarity(self, A: Any, B: Any):
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        if isinstance(B, torch.Tensor):
            B = B.detach().cpu().numpy()
        A = np.atleast_2d(np.asarray(A))
        B = np.atleast_2d(np.asarray(B))

        # normalize when normalize=True
        if getattr(self, "normalize", False):
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

        return A @ B.T

    def similarity_pairwise(self, A: Any, B: Any):
        if isinstance(A, torch.Tensor):
            A = A.detach().cpu().numpy()
        if isinstance(B, torch.Tensor):
            B = B.detach().cpu().numpy()
        A = np.atleast_2d(np.asarray(A))
        B = np.atleast_2d(np.asarray(B))

        # normalize when normalize=True
        if getattr(self, "normalize", False):
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)

        return np.sum(A * B, axis=1)

    def _get_torch_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float32":
            return torch.float32

        dev = self._device_str()
        return torch.float16 if (dev.startswith("cuda") and torch.cuda.is_available()) else torch.float32

    def _lazy_load(self) -> None:
        tokenizer_added_new_token = False

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                revision=self.revision,
                use_fast=True,
            )

            # --- ensure tokenizer has a pad token (Pythia/GPTNeoX often doesn't) ---
            if self._tokenizer.pad_token is None:
                if self._tokenizer.eos_token is not None:
                    # reuse EOS as PAD (preferred: no vocab resize)
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                else:
                    # fallback: create a PAD token if EOS doesn't exist (rare)
                    self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    tokenizer_added_new_token = True

            self._tokenizer.padding_side = "left"  # Match paper's implementation

        if self._model is None:
            torch_dtype = self._get_torch_dtype()
            self._model = AutoModel.from_pretrained(
                self.model_id,
                revision=self.revision,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
            )

            # keep model config consistent
            if getattr(self._model.config, "pad_token_id", None) is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id  # type: ignore[union-attr]

            # only needed if we added a brand new token (not if we reused eos_token)
            if tokenizer_added_new_token:
                self._model.resize_token_embeddings(len(self._tokenizer))  # type: ignore[arg-type]
            else:
                # extra safety: handle any mismatch without assuming how it happened
                try:
                    if self._model.get_input_embeddings().num_embeddings < len(self._tokenizer):
                        self._model.resize_token_embeddings(len(self._tokenizer))
                except Exception:
                    pass

            self._model.to(self._device_str())
            self._model.eval()

            # Cache layer count once model is loaded
            try:
                self._n_transformer_layers = _infer_num_transformer_layers(self._model.config)
            except Exception:
                pass

    def encode(self, sentences: Any, batch_size: int | None = None, **kwargs: Any) -> np.ndarray:
        self._lazy_load()
        assert self._tokenizer is not None
        assert self._model is not None

        bs = int(batch_size or self.batch_size)

        if _is_dataloader(sentences):
            all_embs: List[np.ndarray] = []
            for batch in sentences:  # type: ignore[assignment]
                texts = _normalize_inputs_to_texts(batch)
                if not texts:
                    continue
                all_embs.append(self._encode_texts(texts, bs))
            if not all_embs:
                return np.zeros((0, 0), dtype=np.float32)
            return np.concatenate(all_embs, axis=0)

        texts = _normalize_inputs_to_texts(sentences)
        return self._encode_texts(texts, bs)

    def _encode_texts(self, texts: List[str], bs: int) -> np.ndarray:
        assert self._tokenizer is not None
        assert self._model is not None

        out_chunks: List[np.ndarray] = []

        dev = self._device_str()

        with torch.no_grad():
            for i in range(0, len(texts), bs):
                chunk = texts[i : i + bs]
                toks = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(dev) for k, v in toks.items()}

                outputs = self._model(**toks, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Model did not return hidden_states; cannot do layer sweep.")

                # hidden_states[0] = embeddings, hidden_states[1] = after block0, ...
                idx = self.layer_index + 1
                if idx >= len(hidden_states):
                    raise IndexError(
                        f"layer_index={self.layer_index} out of range for hidden_states len={len(hidden_states)}"
                    )

                hs = hidden_states[idx]  # [B, T, H]

                if self.pooling == "cls":
                    pooled = hs[:, 0, :]
                else:
                    pooled = _mean_pool(hs, toks["attention_mask"])

                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                out_chunks.append(pooled.detach().float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0)