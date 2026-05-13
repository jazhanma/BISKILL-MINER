"""Shared utilities: episode IO, JSON parsing, simple text helpers."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import DATASET_BUILD_CMD, DATASET_SOURCE, EPISODES_PATH
from .schemas import Episode


def load_episodes(path: Optional[Path] = None) -> List[Episode]:
    """Load episodes from disk, validating with the Pydantic schema.

    If the file is missing AND the user explicitly chose a buildable dataset
    (e.g. ``DATASET_SOURCE=droid``), raise a clear, actionable error pointing
    them at the build script. We never silently fall back to mock data.
    """
    p = Path(path) if path is not None else EPISODES_PATH
    if not p.exists():
        is_active_dataset = p.resolve() == Path(EPISODES_PATH).resolve()
        if is_active_dataset and DATASET_BUILD_CMD:
            msg = (
                f"{DATASET_SOURCE.upper()} episodes JSON not found at {p}. "
                f"Run `{DATASET_BUILD_CMD}` first."
            )
            if DATASET_SOURCE == "droid":
                msg += (
                    " (We only ingest a tiny task-metadata subset — the full "
                    "1.7TB DROID dataset is NOT required.)"
                )
            raise FileNotFoundError(msg)
        raise FileNotFoundError(f"Episodes file not found at {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list at {p}, got {type(raw).__name__}")
    return [Episode(**rec) for rec in raw]


def save_episodes(episodes: List[Episode], path: Optional[Path] = None) -> None:
    p = Path(path) if path is not None else EPISODES_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = [ep.model_dump() for ep in episodes]
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of the first JSON object in arbitrary LLM output.

    LLMs frequently wrap JSON in ```json fences or add a trailing comment.
    We try direct parsing, then find the largest balanced {...} substring.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).rstrip("`").strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    candidate = cleaned[start:end]
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z_]+", text.lower())


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))
