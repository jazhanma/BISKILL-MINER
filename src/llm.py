"""LLM wrapper for task decomposition.

Primary path: Qwen2.5-7B-Instruct via Ollama (or any compatible /api/generate
endpoint). Output is *strictly* taxonomy-constrained, validated, and
retried with backoff. Every analysis returns ``LLMRunInfo`` so the UI
can show exactly what happened — no silent degradation.

Fallback path: deterministic, taxonomy-aware rule-based analyzer that
also predicts failure modes from the skill knowledge base. Used only
when the LLM is unreachable or returns unparseable output across all
retries.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import requests

from .config import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SEC,
    OLLAMA_URL,
)
from .schemas import LLMRunInfo, TaskAnalysis
from .taxonomy import (
    COORDINATION_TYPES,
    KEYWORD_TO_COORDINATION,
    KEYWORD_TO_SKILL,
    SKILLS,
    failure_modes_from_coordination,
    failure_modes_from_skills,
    filter_to_known_coordination,
    filter_to_known_skills,
)
from .utils import extract_json_object, normalize_text

log = logging.getLogger("biskill.llm")


SYSTEM_PROMPT = (
    "You are an expert robotics ML researcher who specializes in bimanual manipulation. "
    "Given a natural-language robot task, decompose it into reusable robot skills, "
    "the bimanual coordination patterns it requires, and the most likely failure modes "
    "of a learned policy executing it. "
    "Reason explicitly about which arm stabilizes versus which arm acts, about whether "
    "the action is symmetric or asymmetric, and about contact dynamics. "
    "You must only choose skills and coordination types from the provided taxonomies. "
    "Return strict JSON. No commentary, no markdown fences, no trailing text."
)


_FEW_SHOTS: List[Dict] = [
    {
        "task": "open a jar",
        "json": {
            "task": "open a jar",
            "required_skills": [
                "bimanual_stabilize",
                "hold_and_manipulate",
                "twisting",
                "force_control",
                "opening",
            ],
            "skill_weights": {
                "bimanual_stabilize": 0.95,
                "hold_and_manipulate": 0.85,
                "twisting": 0.95,
                "force_control": 0.80,
                "opening": 0.70,
            },
            "required_coordination": ["stabilize_then_act"],
            "failure_modes": [
                "support arm releases jar before lid breaks free",
                "torque overshoot strips the lid threads",
                "rotation axis misaligned with the jar axis",
            ],
            "explanation": (
                "One arm must brace the jar against the table while the other applies a "
                "regulated torque to the lid. The action is asymmetric: the support arm holds "
                "a static pose, the manipulating arm executes a torque-controlled rotation."
            ),
        },
    },
    {
        "task": "fold a towel in half",
        "json": {
            "task": "fold a towel in half",
            "required_skills": [
                "folding",
                "deformable_object_handling",
                "bimanual_stabilize",
                "sequencing",
            ],
            "skill_weights": {
                "folding": 1.0,
                "deformable_object_handling": 0.9,
                "bimanual_stabilize": 0.6,
                "sequencing": 0.5,
            },
            "required_coordination": ["symmetric_motion", "simultaneous_bimanual"],
            "failure_modes": [
                "fabric slips out of one gripper mid-fold and unfolds",
                "creases land in the wrong location due to bad grip points",
                "asymmetric drift breaks the mirror constraint",
            ],
            "explanation": (
                "Both arms grasp opposite corners of the towel and execute mirrored "
                "trajectories to bring them together. The motion is simultaneous and "
                "symmetric; the dominant risk is grip loss on the soft material."
            ),
        },
    },
]


def _build_prompt(task: str) -> str:
    skills_str = ", ".join(SKILLS)
    coord_str = ", ".join(COORDINATION_TYPES)

    shots_text = "\n\n".join(
        f"Example {i+1}:\nTask: {shot['task']}\nJSON:\n{json.dumps(shot['json'], indent=2)}"
        for i, shot in enumerate(_FEW_SHOTS)
    )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Skill taxonomy (choose only from these): {skills_str}\n"
        f"Coordination taxonomy (choose only from these): {coord_str}\n\n"
        f"Output schema (all fields required):\n"
        f'{{\n'
        f'  "task": "<echo back the task>",\n'
        f'  "required_skills": ["<skill_from_taxonomy>", ...],\n'
        f'  "skill_weights": {{"<skill_from_taxonomy>": <float in [0,1]>}},\n'
        f'  "required_coordination": ["<coordination_from_taxonomy>", ...],\n'
        f'  "failure_modes": ["<concrete failure mode sentence>", ...],\n'
        f'  "explanation": "<one short paragraph reasoning about coordination>"\n'
        f'}}\n\n'
        f"Constraints:\n"
        f"- Pick 4 to 9 required_skills.\n"
        f"- Pick 1 to 4 required_coordination types.\n"
        f"- skill_weights values must be floats in [0, 1] and roughly reflect importance.\n"
        f"- failure_modes must be 3 to 6 concrete, task-specific failure sentences "
        f"  (not generic). Each should name an arm role or contact event.\n"
        f"- Use only the exact lowercase keys from the taxonomies above.\n\n"
        f"{shots_text}\n\n"
        f"Now decompose this task:\nTask: {task}\nJSON:"
    )


def check_ollama(url: Optional[str] = None, model: Optional[str] = None) -> LLMRunInfo:
    """Probe the Ollama tags endpoint to see if our model is loadable."""
    target_url = url or OLLAMA_URL
    target_model = model or OLLAMA_MODEL
    base = target_url.rsplit("/api/", 1)[0]
    try:
        r = requests.get(f"{base}/api/tags", timeout=2.0)
        if r.status_code != 200:
            return LLMRunInfo(
                used_llm=False,
                model=target_model,
                last_error=f"HTTP {r.status_code} from {base}",
                fallback_reason="ollama_http_error",
            )
        tags = r.json().get("models", [])
        names = [t.get("name", "") for t in tags]
        if any(target_model.split(":")[0] in n for n in names):
            return LLMRunInfo(used_llm=True, model=target_model)
        return LLMRunInfo(
            used_llm=False,
            model=target_model,
            last_error=(
                f"Ollama is up but '{target_model}' is not pulled. "
                f"Run: ollama pull {target_model}"
            ),
            fallback_reason="model_not_pulled",
        )
    except requests.RequestException as exc:
        return LLMRunInfo(
            used_llm=False,
            model=target_model,
            last_error=f"Ollama unreachable at {base}: {exc}",
            fallback_reason="ollama_unreachable",
        )


def _call_ollama(prompt: str, model: str, timeout: float) -> Tuple[Optional[str], Optional[str]]:
    """Single Ollama call. Returns (response_text, error_message)."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, "num_predict": 1024},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", ""), None
    except requests.RequestException as exc:
        return None, str(exc)


def _parse_llm_response(task: str, raw: str) -> Optional[TaskAnalysis]:
    obj = extract_json_object(raw)
    if not obj:
        return None
    try:
        skills = filter_to_known_skills(obj.get("required_skills", []) or [])
        if not skills:
            return None
        coords = filter_to_known_coordination(obj.get("required_coordination", []) or [])

        weights_raw = obj.get("skill_weights", {}) or {}
        weights: Dict[str, float] = {}
        for k, v in weights_raw.items():
            k_norm = str(k).strip().lower().replace(" ", "_").replace("-", "_")
            if k_norm in skills:
                try:
                    weights[k_norm] = max(0.0, min(1.0, float(v)))
                except (TypeError, ValueError):
                    continue
        for s in skills:
            weights.setdefault(s, 0.5)

        fm_raw = obj.get("failure_modes", []) or []
        failure_modes = [str(x).strip() for x in fm_raw if str(x).strip()]
        if not failure_modes:
            failure_modes = (
                failure_modes_from_skills(skills)[:4]
                + failure_modes_from_coordination(coords)[:1]
            )

        explanation = str(obj.get("explanation", "")).strip()

        return TaskAnalysis(
            task=task,
            required_skills=skills,
            skill_weights=weights,
            required_coordination=coords or ["asymmetric_motion"],
            failure_modes=failure_modes,
            explanation=explanation
            or "LLM decomposition completed but no explanation provided.",
        )
    except (TypeError, ValueError) as exc:
        log.warning("Failed to parse LLM JSON: %s", exc)
        return None


def _rule_based_skills(task_lc: str) -> Tuple[List[str], List[str]]:
    skill_hits: Dict[str, int] = {}
    coord_hits: Dict[str, int] = {}

    for kw, skills in KEYWORD_TO_SKILL.items():
        if kw in task_lc:
            for s in skills:
                skill_hits[s] = skill_hits.get(s, 0) + 1
    for kw, coords in KEYWORD_TO_COORDINATION.items():
        if kw in task_lc:
            for c in coords:
                coord_hits[c] = coord_hits.get(c, 0) + 1

    if "and" in task_lc.split() or "then" in task_lc.split() or "while" in task_lc:
        skill_hits["sequencing"] = skill_hits.get("sequencing", 0) + 1
        coord_hits["asymmetric_motion"] = coord_hits.get("asymmetric_motion", 0) + 1

    if not skill_hits:
        skill_hits = {"hold_and_manipulate": 1, "precision_placement": 1}
    if not coord_hits:
        coord_hits = {"asymmetric_motion": 1}

    skills_sorted = sorted(skill_hits.items(), key=lambda kv: kv[1], reverse=True)
    coords_sorted = sorted(coord_hits.items(), key=lambda kv: kv[1], reverse=True)
    return [s for s, _ in skills_sorted[:8]], [c for c, _ in coords_sorted[:3]]


def rule_based_analyze(task: str) -> TaskAnalysis:
    task_lc = normalize_text(task)
    skills, coords = _rule_based_skills(task_lc)

    n = max(1, len(skills))
    weights: Dict[str, float] = {
        s: round(1.0 - (i / (n + 1)) * 0.6, 3) for i, s in enumerate(skills)
    }

    failure_modes = (
        failure_modes_from_skills(skills)[:4]
        + failure_modes_from_coordination(coords)[:2]
    )

    explanation = (
        f"Rule-based decomposition (LLM path skipped). Detected skills "
        f"{', '.join(skills)} and coordination patterns "
        f"{', '.join(coords)} from task keywords. "
        f"Failure modes derived from the skill knowledge base in taxonomy.py."
    )
    return TaskAnalysis(
        task=task,
        required_skills=skills,
        skill_weights=weights,
        required_coordination=coords,
        failure_modes=failure_modes,
        explanation=explanation,
    )


def analyze_task(
    task: str,
    use_llm: bool = True,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: int = 2,
) -> Tuple[TaskAnalysis, LLMRunInfo]:
    """Return ``(analysis, run_info)``.

    The LLM is tried up to ``max_retries+1`` times with a small backoff.
    Each attempt is timed; the run_info reports attempt count, total
    duration, last error if any, and the explicit fallback reason if we
    end up using the rule-based path.
    """
    target_model = model or OLLAMA_MODEL
    target_timeout = timeout or OLLAMA_TIMEOUT_SEC

    if not use_llm:
        info = LLMRunInfo(
            used_llm=False,
            model="rule-based",
            fallback_reason="user_disabled_llm",
        )
        return rule_based_analyze(task), info

    prompt = _build_prompt(task)
    started = time.perf_counter()
    last_error: Optional[str] = None
    attempts = 0

    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        raw, err = _call_ollama(prompt, target_model, target_timeout)
        if err:
            last_error = err
            log.warning("Ollama attempt %d/%d failed: %s", attempts, max_retries + 1, err)
        elif raw:
            parsed = _parse_llm_response(task, raw)
            if parsed is not None:
                duration_ms = (time.perf_counter() - started) * 1000.0
                info = LLMRunInfo(
                    used_llm=True,
                    model=target_model,
                    attempts=attempts,
                    duration_ms=duration_ms,
                )
                return parsed, info
            last_error = "LLM response did not parse as taxonomy-valid JSON"
            log.warning("Attempt %d: %s", attempts, last_error)
        if attempt < max_retries:
            time.sleep(0.4 * (attempt + 1))

    duration_ms = (time.perf_counter() - started) * 1000.0
    fallback_reason = (
        "ollama_unreachable" if last_error and ("Connection" in last_error or "timed out" in last_error)
        else "llm_parse_failure"
    )
    info = LLMRunInfo(
        used_llm=False,
        model="rule-based",
        attempts=attempts,
        duration_ms=duration_ms,
        last_error=last_error,
        fallback_reason=fallback_reason,
    )
    return rule_based_analyze(task), info
