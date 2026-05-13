"""Skill-aware episode retrieval with task-specific explanations.

Given a ``TaskAnalysis`` we:
  1. embed a query that combines the raw task with required skills/coords,
  2. fetch FAISS nearest neighbors,
  3. rerank with a small skill/coordination overlap bonus,
  4. attach a structured, task-specific ``match_reason`` for the UI:
       - which skills matched (and the overlap percentage)
       - whether the coordination matched
       - why this episode is useful for *this* target task
"""
from __future__ import annotations

import logging
from typing import List, Optional

from .config import TOP_K_DEFAULT
from .embeddings import embed_text
from .indexer import EpisodeIndex, get_or_build_index
from .schemas import RetrievedEpisode, TaskAnalysis
from .taxonomy import COORDINATION_DESCRIPTIONS, SKILL_DESCRIPTIONS

log = logging.getLogger("biskill.retriever")


SKILL_BONUS = 0.06
COORD_BONUS = 0.05
OVERLAP_CAP = 0.30


def _build_query_text(analysis: TaskAnalysis) -> str:
    return analysis.to_query_text()


def _skill_label(s: str) -> str:
    return s.replace("_", " ")


def _usefulness_for_task(
    target_task: str,
    matched_skills: List[str],
    matched_coord: List[str],
) -> str:
    """Per-episode, task-specific 'why this transfers' sentence."""
    task = target_task.strip().rstrip(".")
    if matched_coord:
        coord = matched_coord[0]
        descr = COORDINATION_DESCRIPTIONS.get(coord, "")
        if matched_skills:
            primary = _skill_label(matched_skills[0])
            return (
                f"Useful for '{task}' because it demonstrates {coord.replace('_', ' ')} — "
                f"{descr.lower().rstrip('.') if descr else 'the coordination pattern this task needs'} — "
                f"and shows {primary} under that constraint."
            )
        return (
            f"Useful for '{task}' because it demonstrates "
            f"{coord.replace('_', ' ')} coordination, which the target task requires."
        )
    if matched_skills:
        primary = _skill_label(matched_skills[0])
        primary_descr = SKILL_DESCRIPTIONS.get(matched_skills[0], "").lower().rstrip(".")
        return (
            f"Useful for '{task}' because it teaches {primary}"
            + (f" — {primary_descr}." if primary_descr else ".")
        )
    return (
        f"Semantically similar to '{task}' based on task language and object overlap, "
        f"but no exact skill or coordination overlap with the required taxonomy items."
    )


def _build_match_reason(
    analysis: TaskAnalysis,
    matched_skills: List[str],
    matched_coord: List[str],
    overlap_pct: float,
    semantic_score: float,
) -> str:
    """Structured explanation: skills + coordination + task-specific usefulness."""
    skills_part = (
        f"covers skills [{', '.join(matched_skills)}] "
        f"({overlap_pct*100:.0f}% of required skills)"
        if matched_skills
        else "no exact skill overlap with the required set"
    )
    coord_part = (
        f"shares coordination [{matched_coord[0]}] with the target task"
        if matched_coord
        else "different coordination pattern from the target"
    )
    usefulness = _usefulness_for_task(analysis.task, matched_skills, matched_coord)
    return (
        f"This episode {skills_part} and {coord_part}. "
        f"Semantic similarity {semantic_score:.2f}. {usefulness}"
    )


def retrieve_relevant_episodes(
    task_analysis: TaskAnalysis,
    top_k: int = TOP_K_DEFAULT,
    index: Optional[EpisodeIndex] = None,
) -> List[RetrievedEpisode]:
    idx = index if index is not None else get_or_build_index()
    if not idx.episodes:
        return []

    query_text = _build_query_text(task_analysis)
    qvec = embed_text(query_text)

    fetch_k = max(top_k * 3, top_k + 10)
    scores, ids = idx.search(qvec, fetch_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    required_skills = set(task_analysis.required_skills)
    required_coord = set(task_analysis.required_coordination)
    n_required = max(1, len(required_skills))

    candidates: List[RetrievedEpisode] = []
    for sem_score, ep_idx in zip(scores, ids):
        if ep_idx < 0 or ep_idx >= len(idx.episodes):
            continue
        ep = idx.episodes[ep_idx]
        ep_skills = set(ep.skills)
        skill_overlap = sorted(required_skills & ep_skills)
        coord_overlap = (
            [ep.coordination_type] if ep.coordination_type in required_coord else []
        )
        overlap_pct = len(skill_overlap) / n_required

        bonus = 0.0
        if required_skills:
            bonus += SKILL_BONUS * overlap_pct
        if coord_overlap:
            bonus += COORD_BONUS
        bonus = min(bonus, OVERLAP_CAP)

        adjusted = float(sem_score) + bonus
        reason = _build_match_reason(
            task_analysis, skill_overlap, coord_overlap, overlap_pct, float(sem_score)
        )
        usefulness = _usefulness_for_task(task_analysis.task, skill_overlap, coord_overlap)
        candidates.append(
            RetrievedEpisode(
                episode=ep,
                similarity_score=adjusted,
                match_reason=reason,
                matched_skills=skill_overlap,
                matched_coordination=coord_overlap,
                skill_overlap_pct=overlap_pct,
                coord_overlap=bool(coord_overlap),
                usefulness_note=usefulness,
            )
        )

    candidates.sort(key=lambda r: r.similarity_score, reverse=True)
    return candidates[:top_k]
