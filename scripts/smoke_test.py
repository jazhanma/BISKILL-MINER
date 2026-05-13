"""End-to-end smoke test for the BiSkill Miner pipeline.

Runs the rule-based path (so it works without Ollama) on the headline
example tasks, builds the FAISS index from the mock dataset, retrieves,
and produces a full TrainingMix including failure_modes, gap_analysis,
and the machine-readable training_config. Prints a research-style
summary so we can sanity-check the demo before launching Streamlit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.indexer import get_or_build_index  # noqa: E402
from src.llm import analyze_task  # noqa: E402
from src.recommender import recommend_training_mix  # noqa: E402
from src.retriever import retrieve_relevant_episodes  # noqa: E402

TASKS = [
    "crack an egg into a pan and add salt",
    "fold a towel in half",
    "open a jar",
    "pour juice into a cup while holding the cup",
    "wipe a table while holding an object steady",
    "assemble two parts together",
]


def _h(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    idx = get_or_build_index()
    print(f"\nFAISS index: {len(idx.episodes)} episodes, dim={idx.dim}")

    for task in TASKS:
        _h(f"TASK: {task}")
        analysis, run = analyze_task(task, use_llm=False)
        print(
            f"  llm.used={run.used_llm} model={run.model} attempts={run.attempts} "
            f"duration_ms={run.duration_ms:.0f} fallback_reason={run.fallback_reason}"
        )
        print(f"  required_skills:        {analysis.required_skills}")
        print(f"  required_coordination:  {analysis.required_coordination}")
        print(f"  failure_modes:")
        for fm in analysis.failure_modes:
            print(f"    - {fm}")

        retrieved = retrieve_relevant_episodes(analysis, top_k=10, index=idx)
        print(f"  retrieved {len(retrieved)} episodes (top 5):")
        for r in retrieved[:5]:
            print(
                f"    - {r.episode.episode_id}  {r.episode.task_name:<48} "
                f"sim={r.similarity_score:.3f}  overlap={r.skill_overlap_pct*100:5.1f}%  "
                f"coord_match={'Y' if r.coord_overlap else 'N'}"
            )
            print(f"        reason: {r.match_reason}")

        mix = recommend_training_mix(task, retrieved, analysis)
        print("  training_mix:")
        for cat in sorted(mix.categories, key=lambda c: c.weight, reverse=True):
            print(f"    {cat.weight*100:5.1f}%  {cat.name}")

        if mix.gap_analysis:
            print("  gap_analysis (uncovered required skills):")
            for g in mix.gap_analysis:
                print(
                    f"    - {g.skill} (importance {g.importance:.2f})\n"
                    f"        why:        {g.why_it_matters}\n"
                    f"        what_fails: {g.what_fails_without_it}\n"
                    f"        fix:        {g.minimal_fix}"
                )
        else:
            print("  gap_analysis: (none — all required skills covered)")

        cfg = mix.training_config
        if cfg is not None:
            print("  training_config:")
            print(f"    sampling_strategy: {cfg.sampling_strategy}")
            print(f"    curriculum:        {' -> '.join(cfg.curriculum)}")
            print(f"    loss_weighting:    "
                  f"{json.dumps({k: round(v, 3) for k, v in cfg.loss_weighting.items()})}")
            print(f"    notes:             {cfg.notes}")


if __name__ == "__main__":
    main()
