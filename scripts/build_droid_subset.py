"""Build ``data/droid_episodes.json`` from a tiny DROID TFDS slice.

We deliberately read **only** the task instruction off each episode. No
images, no proprioception, no trajectories. This keeps the MacBook safe
while still giving the retrieval pipeline real DROID task semantics.

Example
-------
    python scripts/build_droid_subset.py --split "train[:0.01%]" --max-episodes 100
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.droid_loader import HF_DEFAULT_REPO, load_droid_subset  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a metadata-only DROID subset for BiSkill Miner.",
    )
    p.add_argument(
        "--backend",
        choices=["auto", "hf", "tfds"],
        default="auto",
        help=(
            "Source backend. 'hf' streams from HuggingFace (works on Python "
            "3.14, no TF). 'tfds' uses tensorflow-datasets. 'auto' (default) "
            "prefers HF and falls back to TFDS."
        ),
    )
    p.add_argument(
        "--hf-repo",
        default=HF_DEFAULT_REPO,
        help=(
            f"HuggingFace dataset id when backend != 'tfds'. "
            f"Default: {HF_DEFAULT_REPO} (101 episodes, streams in seconds)."
        ),
    )
    p.add_argument(
        "--split",
        default="train[:0.1%]",
        help='Split spec. TFDS: "train[:0.1%%]". HF: "train" (cap via --max-episodes).',
    )
    p.add_argument(
        "--data-dir",
        default="~/datasets/droid",
        help="TFDS data_dir (ignored by HF backend). ~ is expanded.",
    )
    p.add_argument(
        "--max-episodes",
        type=int,
        default=500,
        help="Hard cap on episode count (default 500; start with 100).",
    )
    p.add_argument(
        "--output",
        default="data/droid_episodes.json",
        help="Where to write the JSON list of Episode records.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    backend_str = args.backend
    if backend_str == "hf" or (backend_str == "auto"):
        print(
            f"Loading DROID subset: backend={backend_str} hf_repo={args.hf_repo} "
            f"split={args.split} max_episodes={args.max_episodes}"
        )
    else:
        print(
            f"Loading DROID subset: backend={backend_str} split={args.split} "
            f"data_dir={args.data_dir} max_episodes={args.max_episodes}"
        )
    episodes = load_droid_subset(
        split=args.split,
        data_dir=args.data_dir,
        max_episodes=args.max_episodes,
        output_path=args.output,
        backend=args.backend,
        hf_repo=args.hf_repo,
    )

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    print(f"\nSaved {len(episodes)} episodes -> {out_path}")

    print("\nFirst 5 task names:")
    for ep in episodes[:5]:
        print(f"  - [{ep.episode_id}] {ep.task_name}")

    skill_counts: Counter = Counter()
    coord_counts: Counter = Counter()
    for ep in episodes:
        coord_counts[ep.coordination_type] += 1
        for s in ep.skills:
            skill_counts[s] += 1

    print("\nSkill distribution (BiSkill Miner taxonomy):")
    for k, v in skill_counts.most_common():
        print(f"  {v:4d}  {k}")

    print("\nCoordination distribution:")
    for k, v in coord_counts.most_common():
        print(f"  {v:4d}  {k}")


if __name__ == "__main__":
    main()
