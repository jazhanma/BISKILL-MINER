"""Build ``data/aist_episodes.json`` from the parsed AIST task list.

Pipeline:
    scripts/parse_aist_task_list.py  ->  data/aist_task_list.json
    scripts/build_aist_dataset.py    ->  data/aist_episodes.json

Each AIST task family is collapsed into one BiSkill Miner Episode record,
preserving the AIST task_id in the episode_id (``aist_0073``) and the
quality_score derived from demonstration count + recency.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.loaders.aist import load_aist_episodes  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "data" / "aist_episodes.json"


def main() -> None:
    episodes = load_aist_episodes()
    payload = [ep.model_dump() for ep in episodes]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {len(episodes)} AIST episodes to {OUTPUT_PATH}")

    coord_counts: Counter = Counter()
    skill_counts: Counter = Counter()
    for ep in episodes:
        coord_counts[ep.coordination_type] += 1
        for s in ep.skills:
            skill_counts[s] += 1

    print("\nCoordination distribution (BiSkill Miner labels):")
    for k, v in coord_counts.most_common():
        print(f"  {v:4d}  {k}")
    print("\nSkill distribution (BiSkill Miner labels):")
    for k, v in skill_counts.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
