"""Loader for the AIST Bimanual Manipulation Dataset.

Source: https://aistairc.github.io/aist_bimanip_site/
Citation: Motoda et al., AIST-Bimanual Manipulation, 2025. CC BY 4.0.

AIST publishes 117 task families and 10,705 teleoperated bimanual episodes
collected with the ALOHA dual-arm system. Each task family ships with:
  - task_id, snake_case task_name (e.g. ``fold_blue_towel``)
  - AIST's own taxonomy (one of 5 classes)
  - a skill verb (one of 23)
  - episode count
  - download URL for the per-task zip

We don't need the raw videos / trajectories for BiSkill Miner's retrieval
stage — only the textual task description, the skill labels, and the
coordination type. This loader synthesizes a research-grade ``Episode``
record per AIST task family using two hand-curated mappings:

  - ``AIST_TAXONOMY_TO_COORDINATION``: 5 AIST classes -> our COORDINATION_TYPES
  - ``AIST_SKILL_VERB_TO_SKILLS``: 23 AIST verbs -> our SKILLS

The mappings are taxonomy-validated at module load.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..schemas import Episode
from ..taxonomy import COORDINATION_SET, SKILL_SET


AIST_TAXONOMY_TO_COORDINATION: Dict[str, str] = {
    "Synchronous Bimanual": "simultaneous_bimanual",
    "Asynchronous Bimanual": "asymmetric_motion",
    "Unimanual": "single_arm",
    "independent": "independent_dual_arm",
    "collaboration": "support_and_manipulate",
}


AIST_TAXONOMY_DESCRIPTION: Dict[str, str] = {
    "Synchronous Bimanual": (
        "Both arms act on the same workpiece at the same time with shared timing."
    ),
    "Asynchronous Bimanual": (
        "Arms execute distinct trajectories with different roles (often stabilize-then-act)."
    ),
    "Unimanual": "Only one arm is task-relevant; the other is idle.",
    "independent": "Both arms work, but on independent sub-tasks with no tight coupling.",
    "collaboration": (
        "Human-robot collaboration: one arm provides passive support while the human acts."
    ),
}


AIST_SKILL_VERB_TO_SKILLS: Dict[str, List[str]] = {
    "brush": ["wiping", "tool_use", "bimanual_stabilize", "contact_rich_motion"],
    "close": ["opening", "force_control", "bimanual_stabilize"],
    "find": ["precision_placement", "container_interaction", "force_control"],
    "fit": ["precision_placement", "force_control", "contact_rich_motion"],
    "fold": [
        "folding",
        "deformable_object_handling",
        "bimanual_stabilize",
        "sequencing",
    ],
    "handover": ["bimanual_handoff", "hold_and_manipulate", "precision_placement"],
    "hit": ["contact_rich_motion", "force_control", "tool_use"],
    "hook": ["precision_placement", "force_control", "contact_rich_motion"],
    "insert": [
        "precision_placement",
        "force_control",
        "container_interaction",
        "contact_rich_motion",
    ],
    "lift": ["bimanual_stabilize", "hold_and_manipulate", "force_control"],
    "open": ["opening", "twisting", "force_control", "bimanual_stabilize"],
    "pass": ["bimanual_handoff", "hold_and_manipulate"],
    "peel": [
        "contact_rich_motion",
        "force_control",
        "deformable_object_handling",
        "delicate_grasp",
        "bimanual_stabilize",
    ],
    "pick": ["delicate_grasp", "hold_and_manipulate", "precision_placement"],
    "place": ["precision_placement", "container_interaction", "hold_and_manipulate"],
    "put": ["precision_placement", "container_interaction", "hold_and_manipulate"],
    "remove": ["pulling", "force_control", "hold_and_manipulate"],
    "sliding": ["precision_placement", "force_control", "contact_rich_motion"],
    "spoon": ["tool_use", "container_interaction", "hold_and_manipulate", "stirring"],
    "take": ["pulling", "hold_and_manipulate", "container_interaction"],
    "tape": [
        "tool_use",
        "deformable_object_handling",
        "precision_placement",
        "force_control",
        "bimanual_stabilize",
    ],
    "turn": [
        "deformable_object_handling",
        "folding",
        "bimanual_stabilize",
        "force_control",
    ],
    "wipe": ["wiping", "tool_use", "contact_rich_motion", "bimanual_stabilize"],
}


def _validate_mappings() -> None:
    for k, v in AIST_TAXONOMY_TO_COORDINATION.items():
        if v not in COORDINATION_SET:
            raise ValueError(f"AIST taxonomy '{k}' maps to unknown coordination '{v}'")
    for k, skills in AIST_SKILL_VERB_TO_SKILLS.items():
        bad = [s for s in skills if s not in SKILL_SET]
        if bad:
            raise ValueError(f"AIST skill verb '{k}' maps to unknown skill(s): {bad}")


_validate_mappings()


def _humanize(task_name: str) -> str:
    """Turn ``fold_blue_towel`` into ``"fold blue towel"`` (verb + object phrase)."""
    return task_name.replace("_", " ").strip()


_OBJECT_STOPWORDS = {
    "a", "an", "the", "of", "to", "into", "onto", "from", "for", "and",
    "with", "in", "on", "at", "by", "left", "right", "human", "hold", "arm",
    "both", "hands", "alternating", "moved", "random", "randomly", "placed",
    "fixed", "robot", "marker", "marked", "narrow", "small", "medium", "large",
    "big", "metallic", "clear", "upright", "assist", "places",
    "fix", "parts",
}


_AIST_VERBS = {
    "brush", "close", "find", "fit", "fold", "handover", "hit", "hook", "insert",
    "lift", "open", "pass", "peel", "pick", "place", "put", "remove", "sliding",
    "spoon", "take", "tape", "turn", "wipe",
}


def _extract_objects(task_name: str) -> List[str]:
    """Pull the object phrases out of a snake_case task name."""
    tokens = task_name.split("_")
    objects: List[str] = []
    current: List[str] = []
    for tok in tokens:
        if tok in _AIST_VERBS or tok in _OBJECT_STOPWORDS:
            if current:
                objects.append(" ".join(current))
                current = []
            continue
        current.append(tok)
    if current:
        objects.append(" ".join(current))
    seen: List[str] = []
    for o in objects:
        if o and o not in seen:
            seen.append(o)
    return seen[:5]


def _quality_from_count_and_date(num_episodes: int, date: str) -> float:
    """Heuristic quality score from demonstration volume and recency.

    More demos -> higher base. Newer date -> small recency bonus. Both
    capped so the score remains comparable to the mock dataset's range.
    """
    base = 0.70 + min(0.18, num_episodes / 1500.0)
    year_bonus = 0.0
    m = re.match(r"(\d{4})", date or "")
    if m:
        year = int(m.group(1))
        if year >= 2025:
            year_bonus = 0.03
        elif year >= 2024:
            year_bonus = 0.015
    return round(min(0.95, base + year_bonus), 3)


def _build_description(task: Dict, skills: List[str], coord: str) -> str:
    """Synthesize a research-voice episode description."""
    verb = task["skill_verb"]
    aist_taxo = task["aist_taxonomy"]
    coord_descr = AIST_TAXONOMY_DESCRIPTION.get(aist_taxo, "")
    objects = _extract_objects(task["task_name"])
    obj_phrase = " ".join(objects) if objects else "the workpiece"
    return (
        f"AIST bimanual manipulation demonstration ({aist_taxo.lower()}): "
        f"{verb} {obj_phrase}. "
        f"{coord_descr} "
        f"Recorded with the ALOHA dual-arm system; "
        f"{task['num_episodes']} teleop episodes available."
    ).strip()


def aist_task_to_episode(task: Dict) -> Episode:
    """Convert one AIST task family entry into a BiSkill Miner Episode.

    One AIST task family -> one Episode. This is the right granularity for
    a *task-level retriever* (the user asks for transferable task families,
    not individual trajectories within a family).
    """
    verb = task["skill_verb"]
    if verb not in AIST_SKILL_VERB_TO_SKILLS:
        raise ValueError(f"Unknown AIST skill verb: {verb}")
    skills = list(AIST_SKILL_VERB_TO_SKILLS[verb])
    if task["num_episodes"] >= 100:
        skills.append("sequencing") if "sequencing" not in skills else None
    coord = AIST_TAXONOMY_TO_COORDINATION.get(task["aist_taxonomy"], "asymmetric_motion")
    description = _build_description(task, skills, coord)
    objects = _extract_objects(task["task_name"])
    quality = _quality_from_count_and_date(task["num_episodes"], task.get("date", ""))

    pretty_name = _humanize(task["task_name"])
    return Episode(
        episode_id=f"aist_{task['task_id']:04d}",
        task_name=pretty_name,
        description=description,
        objects=objects or [task["task_name"].split("_")[-1]],
        skills=[s for s in skills if s in SKILL_SET],
        coordination_type=coord,
        quality_score=quality,
    )


def load_aist_episodes(task_list_path: Optional[Path] = None) -> List[Episode]:
    path = Path(task_list_path) if task_list_path else (
        Path(__file__).resolve().parents[2] / "data" / "aist_task_list.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"AIST task list not found at {path}. "
            f"Run `python scripts/parse_aist_task_list.py` first."
        )
    with path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"AIST task list at {path} is empty or malformed")
    return [aist_task_to_episode(t) for t in tasks]
