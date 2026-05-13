"""Global configuration for BiSkill Miner.

Single source of truth for paths, model names, and tunable parameters.
Values can be overridden via environment variables to support both
MacBook MVP runs and the eventual RTX 3060 Ti workstation.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MOCK_EPISODES_PATH: Path = DATA_DIR / "mock_episodes.json"
AIST_EPISODES_PATH: Path = DATA_DIR / "aist_episodes.json"
DROID_EPISODES_PATH: Path = DATA_DIR / "droid_episodes.json"

DATASET_SOURCE: str = os.environ.get("DATASET_SOURCE", "mock").lower()
"""Which corpus to retrieve from. ``mock`` = scripted MVP data,
``aist`` = real AIST Bimanual Manipulation Dataset task families,
``droid`` = metadata subset of the DROID dataset (built locally)."""

_DATASETS = {
    "mock": {
        "episodes_path": MOCK_EPISODES_PATH,
        "label": "Mock (scripted MVP)",
        "url": None,
        "index_prefix": "episodes",
        "build_cmd": None,
        "is_metadata_only": False,
    },
    "aist": {
        "episodes_path": AIST_EPISODES_PATH,
        "label": "AIST Bimanual Manipulation Dataset (117 task families, 10,705 episodes)",
        "url": "https://aistairc.github.io/aist_bimanip_site/",
        "index_prefix": "aist_episodes",
        "build_cmd": "python scripts/build_aist_dataset.py",
        "is_metadata_only": False,
    },
    "droid": {
        "episodes_path": DROID_EPISODES_PATH,
        "label": "DROID metadata subset (task instructions only — not the full 1.7TB)",
        "url": "https://droid-dataset.github.io/",
        "index_prefix": "droid_episodes",
        "build_cmd": (
            'python scripts/build_droid_subset.py --split "train[:0.01%]" '
            "--max-episodes 100"
        ),
        "is_metadata_only": True,
    },
}

if DATASET_SOURCE not in _DATASETS:
    raise ValueError(
        f"Unknown DATASET_SOURCE='{DATASET_SOURCE}'. Choose one of: {list(_DATASETS)}"
    )

DATASET_LABEL: str = _DATASETS[DATASET_SOURCE]["label"]
DATASET_URL = _DATASETS[DATASET_SOURCE]["url"]
DATASET_BUILD_CMD = _DATASETS[DATASET_SOURCE]["build_cmd"]
DATASET_IS_METADATA_ONLY: bool = _DATASETS[DATASET_SOURCE]["is_metadata_only"]
EPISODES_PATH: Path = _DATASETS[DATASET_SOURCE]["episodes_path"]
_INDEX_PREFIX = _DATASETS[DATASET_SOURCE]["index_prefix"]
FAISS_INDEX_PATH: Path = INDEX_DIR / f"{_INDEX_PREFIX}.faiss"
EPISODE_META_PATH: Path = INDEX_DIR / f"{_INDEX_PREFIX}_meta.json"

OLLAMA_URL: str = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT_SEC: float = float(os.environ.get("OLLAMA_TIMEOUT_SEC", "60"))

EMBED_MODEL_NAME: str = os.environ.get(
    "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
EMBED_DEVICE: str = os.environ.get("EMBED_DEVICE", "cpu")

TOP_K_DEFAULT: int = int(os.environ.get("TOP_K_DEFAULT", "20"))

TARGET_TASK_FLOOR: float = 0.30
TARGET_TASK_CEIL: float = 0.55
MIN_CATEGORY_WEIGHT: float = 0.04

USE_LLM_DEFAULT: bool = os.environ.get("USE_LLM", "true").lower() == "true"
