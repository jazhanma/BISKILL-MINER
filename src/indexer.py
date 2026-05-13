"""FAISS index builder and loader for episode embeddings."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np

from .config import EPISODE_META_PATH, EPISODES_PATH, FAISS_INDEX_PATH
from .embeddings import embed_texts, get_embedding_dim
from .schemas import Episode
from .utils import load_episodes

log = logging.getLogger("biskill.indexer")


@dataclass
class EpisodeIndex:
    index: faiss.Index
    episodes: List[Episode]
    dim: int

    def search(self, query_vec: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        return self.index.search(query_vec, min(top_k, len(self.episodes)))


def _build_faiss(vectors: np.ndarray) -> faiss.Index:
    """Build an inner-product index. Vectors must be L2-normalized so IP == cosine."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def build_index(
    episodes_path: Optional[Path] = None,
    index_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> EpisodeIndex:
    ep_path = Path(episodes_path) if episodes_path else EPISODES_PATH
    idx_path = Path(index_path) if index_path else FAISS_INDEX_PATH
    meta_path_ = Path(meta_path) if meta_path else EPISODE_META_PATH

    episodes = load_episodes(ep_path)
    if not episodes:
        raise ValueError(f"No episodes found in {ep_path}")

    log.info("Embedding %d episodes for FAISS index...", len(episodes))
    texts = [ep.to_embedding_text() for ep in episodes]
    vectors = embed_texts(texts)
    if vectors.shape[0] != len(episodes):
        raise RuntimeError("Embedding count does not match episode count")

    index = _build_faiss(vectors)
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))

    meta = {
        "embed_dim": int(vectors.shape[1]),
        "num_episodes": len(episodes),
        "episodes": [ep.model_dump() for ep in episodes],
    }
    with meta_path_.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Index built: %d vectors, dim=%d -> %s", len(episodes), vectors.shape[1], idx_path)
    return EpisodeIndex(index=index, episodes=episodes, dim=int(vectors.shape[1]))


def load_index(
    index_path: Optional[Path] = None,
    meta_path: Optional[Path] = None,
) -> EpisodeIndex:
    idx_path = Path(index_path) if index_path else FAISS_INDEX_PATH
    meta_path_ = Path(meta_path) if meta_path else EPISODE_META_PATH

    if not idx_path.exists() or not meta_path_.exists():
        raise FileNotFoundError(
            f"Index files missing. Expected {idx_path} and {meta_path_}. Run build_index first."
        )

    index = faiss.read_index(str(idx_path))
    with meta_path_.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    episodes = [Episode(**rec) for rec in meta.get("episodes", [])]
    dim = int(meta.get("embed_dim", get_embedding_dim()))
    return EpisodeIndex(index=index, episodes=episodes, dim=dim)


def get_or_build_index() -> EpisodeIndex:
    try:
        return load_index()
    except FileNotFoundError:
        log.info("No FAISS index on disk; building from %s", EPISODES_PATH)
        return build_index()
