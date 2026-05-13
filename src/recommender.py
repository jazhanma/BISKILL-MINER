"""Training-mix recommender + concrete training-config emitter.

Turns retrieved episodes into a fine-tuning data mixture and a
machine-readable ``TrainingConfig`` that can be lifted directly into a
LeRobot / Diffusion Policy / BC-style training loop.

We do three things old-data-aware:
  1. Per required skill, build a bucket from the retrieved episodes.
  2. Allocate weights from importance × evidence × similarity × quality.
  3. For required skills with *zero* coverage, produce a structured
     ``GapAnalysisItem`` (impact + minimal fix), and bump ``loss_weighting``
     on the target_task_demos slice so the policy is forced to learn the
     gap directly from the new demonstrations.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

from .config import MIN_CATEGORY_WEIGHT, TARGET_TASK_CEIL, TARGET_TASK_FLOOR
from .schemas import (
    GapAnalysisItem,
    RetrievedEpisode,
    TaskAnalysis,
    TrainingConfig,
    TrainingMix,
    TrainingMixCategory,
)
from .taxonomy import SKILL_DESCRIPTIONS, SKILL_GAP_KNOWLEDGE

log = logging.getLogger("biskill.recommender")


MAX_REPRESENTATIVE_EPISODES_PER_CATEGORY = 6
MAX_EXTRA_CATEGORIES = 6


def _skill_label(skill: str) -> str:
    return skill.replace("_", " ")


def _bucket_episodes_by_skill(
    retrieved: List[RetrievedEpisode],
    required_skills: List[str],
) -> Dict[str, List[RetrievedEpisode]]:
    buckets: Dict[str, List[RetrievedEpisode]] = defaultdict(list)
    for r in retrieved:
        for skill in r.episode.skills:
            if skill in required_skills:
                buckets[skill].append(r)
    for skill in buckets:
        buckets[skill].sort(key=lambda r: r.similarity_score, reverse=True)
    return buckets


def _coverage_counts(
    retrieved: List[RetrievedEpisode],
    required_skills: List[str],
) -> Dict[str, int]:
    counts = {s: 0 for s in required_skills}
    for r in retrieved:
        for s in r.episode.skills:
            if s in counts:
                counts[s] += 1
    return counts


def _category_score(
    skill: str,
    bucket: List[RetrievedEpisode],
    skill_weights: Dict[str, float],
) -> float:
    if not bucket:
        return 0.0
    importance = max(0.05, float(skill_weights.get(skill, 0.5)))
    evidence = min(1.0, len(bucket) / 5.0)
    avg_sim = sum(max(0.0, r.similarity_score) for r in bucket) / len(bucket)
    avg_quality = sum(r.episode.quality_score for r in bucket) / len(bucket)
    return importance * (0.55 * evidence + 0.30 * avg_sim + 0.15 * avg_quality)


def _normalize_with_target(
    raw_weights: Dict[str, float],
    target_share: float,
) -> Dict[str, float]:
    extras = {k: v for k, v in raw_weights.items() if k != "target_task_demos"}
    extras_sum = sum(extras.values())
    if extras_sum <= 0:
        return {"target_task_demos": 1.0}
    rest_share = max(0.0, 1.0 - target_share)
    out = {"target_task_demos": target_share}
    for k, v in extras.items():
        out[k] = (v / extras_sum) * rest_share
    return out


def _compute_target_share(
    coverage: Dict[str, int],
    required_skills: List[str],
    has_uncovered: bool,
) -> float:
    if not required_skills:
        return TARGET_TASK_CEIL
    covered = sum(1 for s in required_skills if coverage.get(s, 0) > 0)
    coverage_ratio = covered / max(1, len(required_skills))
    span = TARGET_TASK_CEIL - TARGET_TASK_FLOOR
    base = TARGET_TASK_FLOOR + (1.0 - coverage_ratio) * span
    if has_uncovered:
        base = min(TARGET_TASK_CEIL, base + 0.05)
    return base


def _build_gap_analysis(
    uncovered_skills: List[str],
    skill_weights: Dict[str, float],
) -> List[GapAnalysisItem]:
    items: List[GapAnalysisItem] = []
    for skill in uncovered_skills:
        kb = SKILL_GAP_KNOWLEDGE.get(skill, {})
        items.append(
            GapAnalysisItem(
                skill=skill,
                importance=float(skill_weights.get(skill, 0.5)),
                why_it_matters=kb.get(
                    "why",
                    f"This skill is required by the target task but no entry exists in the gap knowledge base for '{skill}'.",
                ),
                what_fails_without_it=kb.get(
                    "what_fails",
                    "the policy will fail at the sub-step that depends on this skill",
                ),
                minimal_fix=kb.get(
                    "minimal_fix",
                    f"add 15-25 demos that explicitly exercise {_skill_label(skill)}",
                ),
            )
        )
    items.sort(key=lambda i: i.importance, reverse=True)
    return items


def _build_training_config(
    recommended_mix: Dict[str, float],
    categories: List[TrainingMixCategory],
    gap_analysis: List[GapAnalysisItem],
    skill_weights: Dict[str, float],
) -> TrainingConfig:
    """Convert the percentage mix into a concrete training config.

    - sampling_strategy: weighted random sampling per category, with a
      curriculum stage that starts on the gap-heavy slice if any.
    - loss_weighting: per-category loss multiplier. The ``target_task_demos``
      slice is up-weighted when uncovered skills exist so the policy is
      forced to learn the gap directly from new demonstrations.
    - per_category_episodes: explicit episode IDs to sample from each bucket.
    - curriculum: ordering by importance × under-coverage; uncovered skills
      train first inside the target_task_demos block, then high-importance
      covered skills, then the rest.
    """
    per_category_episodes: Dict[str, List[str]] = {
        cat.name: list(cat.representative_episode_ids) for cat in categories
    }

    loss_weighting: Dict[str, float] = {name: 1.0 for name in recommended_mix}
    if gap_analysis:
        loss_weighting["target_task_demos"] = round(
            1.0 + 0.15 * min(4, len(gap_analysis)), 3
        )
        for cat in categories:
            if cat.name == "target_task_demos":
                continue
            imp = float(skill_weights.get(cat.name, 0.5))
            loss_weighting[cat.name] = round(0.85 + 0.30 * imp, 3)

    if gap_analysis:
        sampling_strategy = (
            "curriculum_then_weighted_random: stage 1 trains on target_task_demos with "
            "gap skills heavily sampled; stage 2 mixes all categories by recommended_mix "
            "with weighted random sampling."
        )
    else:
        sampling_strategy = (
            "weighted_random_per_category: each minibatch is sampled across categories "
            "according to recommended_mix; episodes within a category are uniform."
        )

    sorted_categories = sorted(
        [c for c in categories if c.name != "target_task_demos"],
        key=lambda c: -float(skill_weights.get(c.name, 0.0)),
    )
    curriculum: List[str] = ["target_task_demos"] + [c.name for c in sorted_categories]

    notes_parts: List[str] = [
        f"target_task_demos floor was sized to {recommended_mix.get('target_task_demos', 0)*100:.0f}% based on coverage of required skills.",
    ]
    if gap_analysis:
        gap_names = ", ".join(g.skill for g in gap_analysis)
        notes_parts.append(
            f"Uncovered skills [{gap_names}] cannot be supplied by old data; "
            f"target_task_demos loss is up-weighted to force the policy to learn them directly."
        )
    else:
        notes_parts.append(
            "All required skills are represented in retrieved old episodes; "
            "no curriculum stage is required."
        )

    return TrainingConfig(
        dataset_mix=recommended_mix,
        sampling_strategy=sampling_strategy,
        loss_weighting=loss_weighting,
        per_category_episodes=per_category_episodes,
        curriculum=curriculum,
        notes=" ".join(notes_parts),
    )


def _build_explanation(
    task: str,
    categories: List[TrainingMixCategory],
    gap_analysis: List[GapAnalysisItem],
    target_share: float,
) -> str:
    if not categories:
        return (
            f"No old episodes covered the required skills for '{task}'. "
            f"Recommend using only freshly collected target-task demos until "
            f"adjacent skills are available."
        )

    bullets: List[str] = []
    for cat in categories:
        if cat.name == "target_task_demos":
            continue
        bullets.append(
            f"- **{_skill_label(cat.name)} ({cat.weight*100:.0f}%)**: {cat.rationale}"
        )

    gap_block = ""
    if gap_analysis:
        gap_lines = []
        for g in gap_analysis:
            gap_lines.append(
                f"  - **{_skill_label(g.skill)}** (importance {g.importance:.2f}): "
                f"{g.why_it_matters} Without it, {g.what_fails_without_it}. "
                f"Minimal fix: {g.minimal_fix}."
            )
        gap_block = (
            "\n\n**Coverage gaps** — required skills not present in retrieved old data:\n"
            + "\n".join(gap_lines)
            + "\n\nThe target-task demos must teach these directly; no adjacent slice will substitute."
        )

    return (
        f"Mix for **{task}** allocates {target_share*100:.0f}% to target-task demos. "
        f"The remaining budget is spread across reusable old episodes that already "
        f"teach the required coordination patterns:\n\n"
        + "\n".join(bullets)
        + gap_block
    )


def recommend_training_mix(
    task: str,
    retrieved_episodes: List[RetrievedEpisode],
    task_analysis: TaskAnalysis,
) -> TrainingMix:
    required_skills = list(task_analysis.required_skills)
    skill_weights = dict(task_analysis.skill_weights)

    coverage = _coverage_counts(retrieved_episodes, required_skills)
    buckets = _bucket_episodes_by_skill(retrieved_episodes, required_skills)

    skill_scores: Dict[str, float] = {
        skill: _category_score(skill, buckets.get(skill, []), skill_weights)
        for skill in required_skills
    }
    ranked = [
        (skill, score) for skill, score in skill_scores.items()
        if score > 0 and buckets.get(skill)
    ]
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    ranked = ranked[:MAX_EXTRA_CATEGORIES]

    uncovered = [
        s for s in required_skills
        if coverage.get(s, 0) == 0 and skill_weights.get(s, 0) > 0
    ]
    gap_analysis = _build_gap_analysis(uncovered, skill_weights)

    target_share = _compute_target_share(coverage, required_skills, bool(uncovered))

    raw_weights: Dict[str, float] = {"target_task_demos": target_share}
    for skill, score in ranked:
        raw_weights[skill] = score

    normalized = _normalize_with_target(raw_weights, target_share)

    categories: List[TrainingMixCategory] = [
        TrainingMixCategory(
            name="target_task_demos",
            weight=normalized.get("target_task_demos", target_share),
            rationale=(
                f"Fresh demonstrations of '{task}' anchor the policy on the exact object "
                f"set, scene layout, and success criterion. No old episode can substitute "
                f"for this slice; if uncovered skills exist, this slice must teach them."
            ),
            representative_episode_ids=[],
        )
    ]
    floored: Dict[str, float] = {"target_task_demos": normalized["target_task_demos"]}

    for skill, _score in ranked:
        weight = normalized.get(skill, 0.0)
        if weight < MIN_CATEGORY_WEIGHT:
            continue
        bucket = buckets.get(skill, [])
        rep_ids = [
            r.episode.episode_id
            for r in bucket[:MAX_REPRESENTATIVE_EPISODES_PER_CATEGORY]
        ]
        descr = SKILL_DESCRIPTIONS.get(skill, "")
        rationale = (
            f"{descr} {len(bucket)} retrieved episodes teach this skill, including: "
            f"{', '.join(r.episode.task_name for r in bucket[:3])}."
        )
        categories.append(
            TrainingMixCategory(
                name=skill,
                weight=weight,
                rationale=rationale,
                representative_episode_ids=rep_ids,
            )
        )
        floored[skill] = weight

    total = sum(floored.values())
    if total > 0:
        floored = {k: v / total for k, v in floored.items()}
        for cat in categories:
            cat.weight = floored.get(cat.name, cat.weight)

    explanation = _build_explanation(
        task, categories, gap_analysis, floored.get("target_task_demos", target_share)
    )

    training_config = _build_training_config(
        recommended_mix=floored,
        categories=categories,
        gap_analysis=gap_analysis,
        skill_weights=skill_weights,
    )

    return TrainingMix(
        target_task=task,
        recommended_mix=floored,
        categories=categories,
        selected_episodes=retrieved_episodes,
        coverage=coverage,
        uncovered_skills=uncovered,
        gap_analysis=gap_analysis,
        training_config=training_config,
        explanation=explanation,
    )
