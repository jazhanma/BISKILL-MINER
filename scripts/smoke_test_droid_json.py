"""Smoke test for the DROID JSON path — no DROID download required.

Strategy
--------
1. Write a tiny fake ``data/droid_episodes.json`` (3 records matching the
   shape that ``convert_droid_sample_to_episode`` would have produced).
2. In a child Python process with ``DATASET_SOURCE=droid``, build the
   FAISS index, run the analyzer (rule-based), retrieve, and recommend.
3. Restore: delete the fake JSON (or restore a backup if one existed)
   and drop the temporary droid index so the user's environment is
   untouched.

Purpose: prove the app can ingest a DROID-shaped JSON end-to-end on a
MacBook *without* installing tensorflow or pulling a single byte of
DROID.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


_FAKE_EPISODES = [
    {
        "episode_id": "droid_000001",
        "task_name": "pick up the red block and place it on the shelf",
        "description": "pick up the red block and place it on the shelf",
        "objects": [],
        "skills": ["delicate_grasp", "hold_and_manipulate", "precision_placement"],
        "coordination_type": "single_arm",
        "quality_score": 0.85,
    },
    {
        "episode_id": "droid_000002",
        "task_name": "open the cabinet drawer",
        "description": "open the cabinet drawer",
        "objects": [],
        "skills": ["opening", "force_control", "bimanual_stabilize"],
        "coordination_type": "asymmetric_motion",
        "quality_score": 0.85,
    },
    {
        "episode_id": "droid_000003",
        "task_name": "wipe the table with a sponge",
        "description": "wipe the table with a sponge",
        "objects": [],
        "skills": ["wiping", "tool_use", "contact_rich_motion"],
        "coordination_type": "single_arm",
        "quality_score": 0.85,
    },
]


_CHILD_SCRIPT = r"""
import sys
sys.path.insert(0, '.')
from src.indexer import get_or_build_index
from src.llm import analyze_task
from src.retriever import retrieve_relevant_episodes
from src.recommender import recommend_training_mix
from src.config import DATASET_SOURCE, EPISODES_PATH

assert DATASET_SOURCE == 'droid', DATASET_SOURCE
assert EPISODES_PATH.name == 'droid_episodes.json', EPISODES_PATH.name

idx = get_or_build_index()
task = 'wipe the table with a sponge'
analysis, _llm = analyze_task(task, use_llm=False)
retrieved = retrieve_relevant_episodes(analysis, top_k=3, index=idx)
mix = recommend_training_mix(task, retrieved, analysis)

print(f'index_size={len(idx.episodes)}')
print(f'top_ids={[e.episode.episode_id for e in retrieved]}')
print(f'mix_keys={list(mix.recommended_mix.keys())[:5]}')
print(f'training_config_present={mix.training_config is not None}')
"""


def main() -> int:
    droid_json = REPO_ROOT / "data" / "droid_episodes.json"
    droid_index = REPO_ROOT / "data" / "index" / "droid_episodes.faiss"
    droid_meta = REPO_ROOT / "data" / "index" / "droid_episodes_meta.json"

    backup: Path | None = None
    if droid_json.exists():
        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="droid_backup_")
        os.close(fd)
        backup = Path(tmp)
        shutil.copy2(droid_json, backup)
        print(f"Backed up existing {droid_json.name} -> {backup}")

    droid_json.parent.mkdir(parents=True, exist_ok=True)
    with droid_json.open("w", encoding="utf-8") as f:
        json.dump(_FAKE_EPISODES, f, indent=2)
    print(f"Wrote fake DROID JSON with {len(_FAKE_EPISODES)} records.")

    droid_index.unlink(missing_ok=True)
    droid_meta.unlink(missing_ok=True)

    env = os.environ.copy()
    env["DATASET_SOURCE"] = "droid"
    env["USE_LLM"] = "false"
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))
    env.setdefault("TRANSFORMERS_CACHE", str(REPO_ROOT / ".hf_cache"))

    rc = 0
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        print("\n--- child stdout ---")
        print(result.stdout)
        if result.stderr.strip():
            print("--- child stderr ---", file=sys.stderr)
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            print("\nDROID JSON smoke test FAILED (child non-zero exit).")
            rc = result.returncode or 1
        elif "index_size=3" not in result.stdout:
            print("\nDROID JSON smoke test FAILED (unexpected index size).")
            rc = 1
        elif "droid_" not in result.stdout:
            print("\nDROID JSON smoke test FAILED (no droid_* episode ids retrieved).")
            rc = 1
        elif "training_config_present=True" not in result.stdout:
            print("\nDROID JSON smoke test FAILED (no training_config emitted).")
            rc = 1
        else:
            print("\nDROID JSON smoke test PASSED")
    finally:
        droid_index.unlink(missing_ok=True)
        droid_meta.unlink(missing_ok=True)
        if backup is not None:
            shutil.move(str(backup), str(droid_json))
            print(f"Restored original {droid_json.name} from backup.")
        else:
            droid_json.unlink(missing_ok=True)
            print(f"Removed fake {droid_json.name}.")

    return rc


if __name__ == "__main__":
    sys.exit(main())
