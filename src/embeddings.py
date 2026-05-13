"""Text embedding wrapper.

MVP uses ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, ~80MB) so the
whole pipeline runs on a MacBook with no GPU. Swappable to
``Qwen3-Embedding-0.6B`` later via ``EMBED_MODEL_NAME`` env var.
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from .config import EMBED_DEVICE, EMBED_MODEL_NAME

log = logging.getLogger("biskill.embeddings")


class _EmbedderSingleton:
    _lock = threading.Lock()
    _model = None
    _model_name: Optional[str] = None
    _dim: Optional[int] = None

    @classmethod
    def get(cls, model_name: str = EMBED_MODEL_NAME, device: str = EMBED_DEVICE):
        with cls._lock:
            if cls._model is None or cls._model_name != model_name:
                from sentence_transformers import SentenceTransformer

                log.info("Loading embedding model %s on %s", model_name, device)
                cls._model = SentenceTransformer(model_name, device=device)
                cls._model_name = model_name
                cls._dim = int(cls._model.get_sentence_embedding_dimension())
            return cls._model

    @classmethod
    def dim(cls) -> int:
        if cls._dim is None:
            cls.get()
        assert cls._dim is not None
        return cls._dim


def get_embedding_dim() -> int:
    return _EmbedderSingleton.dim()


def embed_text(text: str) -> np.ndarray:
    """Embed a single string, L2-normalized for cosine / inner-product search."""
    if not isinstance(text, str):
        text = str(text)
    model = _EmbedderSingleton.get()
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )[0]
    return vec.astype(np.float32, copy=False)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a batch of strings; returns shape (N, dim) float32, L2-normalized."""
    if not texts:
        return np.zeros((0, get_embedding_dim()), dtype=np.float32)
    model = _EmbedderSingleton.get()
    arr = model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return arr.astype(np.float32, copy=False)
