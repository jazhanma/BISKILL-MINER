"""Pydantic schemas shared across the BiSkill Miner pipeline."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Episode(BaseModel):
    """A single bimanual robot demonstration episode."""

    episode_id: str
    task_name: str
    description: str
    objects: List[str] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    coordination_type: str
    quality_score: float = Field(ge=0.0, le=1.0)

    def to_embedding_text(self) -> str:
        skills = ", ".join(self.skills) if self.skills else "none"
        objects = ", ".join(self.objects) if self.objects else "none"
        return (
            f"{self.task_name}. {self.description} "
            f"objects: {objects}. "
            f"skills: {skills}. "
            f"coordination: {self.coordination_type}"
        )


class TaskAnalysis(BaseModel):
    """LLM-produced decomposition of a natural-language robot task.

    ``failure_modes`` enumerates concrete ways the bimanual policy is most
    likely to fail on this task. They are produced by the LLM and / or
    derived from the required skills via the taxonomy knowledge base.
    """

    task: str
    required_skills: List[str] = Field(default_factory=list)
    skill_weights: Dict[str, float] = Field(default_factory=dict)
    required_coordination: List[str] = Field(default_factory=list)
    failure_modes: List[str] = Field(default_factory=list)
    explanation: str = ""

    @field_validator("skill_weights")
    @classmethod
    def _clip_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        return {k: max(0.0, float(val)) for k, val in v.items()}

    def normalized_skill_weights(self) -> Dict[str, float]:
        if not self.skill_weights:
            return {}
        total = sum(self.skill_weights.values())
        if total <= 0:
            return {k: 0.0 for k in self.skill_weights}
        return {k: float(v) / total for k, v in self.skill_weights.items()}

    def to_query_text(self) -> str:
        skills = ", ".join(self.required_skills) if self.required_skills else "none"
        coord = (
            ", ".join(self.required_coordination)
            if self.required_coordination
            else "none"
        )
        return (
            f"Target task: {self.task}. "
            f"Required skills: {skills}. "
            f"Required coordination: {coord}. "
            f"Notes: {self.explanation}"
        )


class RetrievedEpisode(BaseModel):
    """Episode retrieved from the FAISS index, with structured explainability."""

    episode: Episode
    similarity_score: float
    match_reason: str = ""
    matched_skills: List[str] = Field(default_factory=list)
    matched_coordination: List[str] = Field(default_factory=list)
    skill_overlap_pct: float = 0.0
    coord_overlap: bool = False
    usefulness_note: str = ""


class TrainingMixCategory(BaseModel):
    """One bucket of the suggested fine-tuning mixture."""

    name: str
    weight: float
    rationale: str
    representative_episode_ids: List[str] = Field(default_factory=list)


class GapAnalysisItem(BaseModel):
    """One required skill that the retrieved old data does NOT cover."""

    skill: str
    importance: float
    why_it_matters: str
    what_fails_without_it: str
    minimal_fix: str


class TrainingConfig(BaseModel):
    """Concrete, machine-readable post-training configuration.

    Designed to be lifted directly into a LeRobot / Diffusion Policy /
    BC-style training loop. ``loss_weighting`` boosts skills that are
    important *and* under-represented; ``curriculum`` defines a stage
    ordering for staged fine-tuning.
    """

    dataset_mix: Dict[str, float]
    sampling_strategy: str
    loss_weighting: Dict[str, float] = Field(default_factory=dict)
    per_category_episodes: Dict[str, List[str]] = Field(default_factory=dict)
    curriculum: List[str] = Field(default_factory=list)
    notes: str = ""


class TrainingMix(BaseModel):
    """Final recommended fine-tuning mixture for a target task."""

    target_task: str
    recommended_mix: Dict[str, float]
    categories: List[TrainingMixCategory] = Field(default_factory=list)
    selected_episodes: List[RetrievedEpisode] = Field(default_factory=list)
    coverage: Dict[str, int] = Field(default_factory=dict)
    uncovered_skills: List[str] = Field(default_factory=list)
    gap_analysis: List[GapAnalysisItem] = Field(default_factory=list)
    training_config: Optional[TrainingConfig] = None
    explanation: str = ""

    @field_validator("recommended_mix")
    @classmethod
    def _check_mix(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            return v
        total = sum(v.values())
        if total <= 0:
            return v
        return {k: float(w) / total for k, w in v.items()}


class LLMRunInfo(BaseModel):
    """Telemetry about how the task was decomposed.

    Populated for every analysis so the UI can show *exactly* whether the
    LLM ran, how long it took, how many retries it used, and why it
    failed if it did. No silent fallbacks.
    """

    used_llm: bool
    model: str
    attempts: int = 0
    duration_ms: float = 0.0
    last_error: Optional[str] = None
    fallback_reason: Optional[str] = None


class AnalysisBundle(BaseModel):
    """Top-level result returned to the UI."""

    task: str
    analysis: TaskAnalysis
    retrieved: List[RetrievedEpisode]
    training_mix: TrainingMix
    llm: LLMRunInfo
