"""Streamlit UI sections for BiSkill Miner — plain-language, tabbed layout."""
from __future__ import annotations

import json
from typing import Callable, List, Optional

import pandas as pd
import streamlit as st

from .schemas import AnalysisBundle, RetrievedEpisode, TaskAnalysis, TrainingMix
from .taxonomy import COORDINATION_DESCRIPTIONS


def html(snippet: str) -> None:
    st.markdown(snippet, unsafe_allow_html=True)


def card_open(title: str, color: str = "cyan", sub: str = "") -> None:
    sub_html = f'<span class="bsm-card-sub">{sub}</span>' if sub else ""
    html(
        f"""
        <div class="bsm-card bsm-card-{color}">
            <div class="bsm-card-head">
                <div class="bsm-card-title"><span class="bsm-prefix">//</span>{title}</div>
                {sub_html}
            </div>
        """
    )


def card_close() -> None:
    html("</div>")


def tag_row(
    items: List[str], color: str = "cyan", meta: Optional[dict] = None
) -> None:
    if not items:
        return
    cls = f"bsm-tag bsm-tag-{color}" if color != "cyan" else "bsm-tag"
    pieces: List[str] = []
    for it in items:
        label = it.replace("_", " ")
        meta_html = ""
        if meta and it in meta:
            meta_html = f'<span class="bsm-tag-meta">{meta[it]}</span>'
        pieces.append(f'<span class="{cls}">{label}{meta_html}</span>')
    html('<div class="bsm-tags">' + "".join(pieces) + "</div>")


def status_chip(label: str, value: str, kind: str = "neutral") -> str:
    cls = {
        "ok": "bsm-chip-ok",
        "warn": "bsm-chip-warn",
        "alert": "bsm-chip-alert",
        "neutral": "",
    }.get(kind, "")
    return (
        f'<span class="bsm-chip {cls}"><span class="dot"></span>'
        f'<span>{label}</span><span style="color:var(--text);opacity:0.85">{value}</span></span>'
    )


def _llm_plain_sentence(bundle: AnalysisBundle) -> str:
    r = bundle.llm
    if r.used_llm:
        return (
            f"Task understanding used the **{r.model}** model (~{r.duration_ms:.0f} ms). "
            "Numbers below come from that analysis."
        )
    if r.fallback_reason == "user_disabled_llm":
        return (
            "AI model is **turned off** (see sidebar). The app used built-in rules "
            "to guess skills — good for a quick check; turn on Ollama for richer output."
        )
    return (
        f"The AI model was **not available** ({r.fallback_reason or 'unknown'}). "
        "Results use the same built-in rules. Start Ollama and enable it in the sidebar for full quality."
    )


def render_overview(
    bundle: AnalysisBundle, dataset_source: str, indexed_count: int
) -> None:
    """High-level dashboard: task, metrics, LLM note, teaser text."""
    html(
        f'<div class="bsm-task-quote"><span class="bsm-task-quote-label">Your task</span>'
        f"<blockquote>{bundle.task}</blockquote></div>"
    )

    n_skills = len(bundle.analysis.required_skills)
    n_coord = len(bundle.analysis.required_coordination)
    n_fail = len(bundle.analysis.failure_modes)
    n_gaps = len(bundle.training_mix.gap_analysis)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Skills identified", str(n_skills), help="Rough behaviors the robot must do for this task.")
    m2.metric("Arm patterns", str(n_coord), help="How the two arms need to work together.")
    m3.metric("Risk scenarios", str(n_fail), help="Ways a trained policy might realistically fail.")
    m4.metric("Missing from data", str(n_gaps), help="Skills the indexed dataset may not teach; you may need new demos.")
    best_sim = (
        round(bundle.retrieved[0].similarity_score, 2) if bundle.retrieved else 0.0
    )
    m5.metric("Best episode match", f"{best_sim:.2f}", help="Similarity of the closest stored episode (0–1 scale).")

    st.caption(
        f"Searching within **{indexed_count:,}** indexed episodes · dataset: **{dataset_source}**"
    )

    if bundle.llm.used_llm:
        st.success(_llm_plain_sentence(bundle))
    elif bundle.llm.fallback_reason == "user_disabled_llm":
        st.info(_llm_plain_sentence(bundle))
    else:
        st.warning(_llm_plain_sentence(bundle))

    st.markdown("**In short**")
    teaser = bundle.training_mix.explanation or ""
    if len(teaser) > 600:
        teaser = teaser[:600].rsplit(" ", 1)[0] + "…"
    st.markdown(teaser if teaser else "Open the other tabs for skills, similar episodes, and training suggestions.")

    st.markdown("")
    st.info(
        "Tip: start with **Skills & risks** to sanity-check the breakdown, then **Similar episodes** "
        "to see what old data might transfer, then **Training plan** for mix and gaps."
    )


def render_briefing(analysis: TaskAnalysis) -> None:
    card_open(
        "Skills & how arms work together",
        "cyan",
        "What the robot needs to do · coordination · things that often go wrong",
    )
    st.caption(
        "These labels come from a fixed skill list. They help match your new task to old demonstrations."
    )

    cols = st.columns([1.1, 0.9, 1.1])

    with cols[0]:
        st.markdown("**Skills**")
        norm = analysis.normalized_skill_weights()
        weight_meta = (
            {k: f"{norm.get(k, 0.0)*100:.0f}%" for k in analysis.required_skills}
            if norm
            else None
        )
        tag_row(analysis.required_skills, color="cyan", meta=weight_meta)
        if not analysis.required_skills:
            st.caption("No skills listed — try rephrasing the task or enable the AI model.")
        elif analysis.skill_weights:
            with st.expander("Show raw skill weights (advanced)", expanded=False):
                df = (
                    pd.DataFrame(
                        {
                            "Skill": [s.replace("_", " ") for s in analysis.skill_weights.keys()],
                            "Weight (raw)": [round(v, 3) for v in analysis.skill_weights.values()],
                            "Share": [
                                f"{norm.get(k, 0.0)*100:.1f}%"
                                for k in analysis.skill_weights.keys()
                            ],
                        }
                    )
                    .sort_values("Weight (raw)", ascending=False)
                    .reset_index(drop=True)
                )
                st.dataframe(df, use_container_width=True, hide_index=True)

    with cols[1]:
        st.markdown("**Two-arm coordination**")
        tag_row(analysis.required_coordination, color="brass")
        if analysis.required_coordination:
            for coord in analysis.required_coordination:
                descr = COORDINATION_DESCRIPTIONS.get(coord, "")
                if descr:
                    st.caption(f"› {descr}")
        else:
            st.caption("No coordination pattern detected from the task text.")

    with cols[2]:
        st.markdown("**What might go wrong (policy risks)**")
        if not analysis.failure_modes:
            st.caption("No failure scenarios listed for this run.")
        else:
            lis = "".join(f"<li>{fm}</li>" for fm in analysis.failure_modes)
            html(f'<ul class="bsm-wire-list">{lis}</ul>')

    card_close()


def render_salvage(retrieved: List[RetrievedEpisode]) -> None:
    card_open(
        "Similar past episodes",
        "lime",
        f"Up to {len(retrieved)} closest matches from your indexed library",
    )
    st.caption(
        "Match score blends text similarity with overlapping skills. “Skill match %” is overlap with *this* task’s skills."
    )
    if not retrieved:
        st.info("No episodes in the index, or nothing scored as similar. Check your dataset and embeddings.")
        card_close()
        return

    top3 = retrieved[:3]
    mcols = st.columns(len(top3))
    for c, r in zip(mcols, top3):
        with c:
            st.metric(
                label=r.episode.episode_id.replace("_", " ").title()[:24],
                value=f"{r.skill_overlap_pct*100:.0f}% skill overlap",
                delta=f"Score {r.similarity_score:.2f} · pattern {'match' if r.coord_overlap else 'diff'}",
                help="Higher score and overlap usually mean more transferable behavior.",
            )
            st.caption(r.episode.task_name[:120] + ("…" if len(r.episode.task_name) > 120 else ""))

    rows = [
        {
            "Episode": r.episode.episode_id,
            "Summary": r.episode.task_name,
            "Coordination style": r.episode.coordination_type.replace("_", " "),
            "Match score": round(r.similarity_score, 3),
            "Skill match %": round(r.skill_overlap_pct * 100, 1),
            "Pattern match": "Yes" if r.coord_overlap else "No",
            "Overlap skills": ", ".join(r.matched_skills),
            "Quality": r.episode.quality_score,
        }
        for r in retrieved
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("Why each top episode was picked (plain English)", expanded=False):
        for r in retrieved[:8]:
            tn = r.episode.task_name
            if len(tn) > 100:
                tn = tn[:100] + "…"
            st.markdown(f"**{r.episode.episode_id}** — *{tn}*")
            if r.episode.description:
                st.caption(r.episode.description[:300])
            st.markdown(r.match_reason)
            st.divider()

    card_close()


def render_mix(mix: TrainingMix) -> None:
    card_open(
        "How to mix your training data",
        "cyan",
        "Suggested share of each type of demonstration when fine-tuning",
    )
    if not mix.recommended_mix:
        st.info("No mix produced — check that retrieval returned episodes.")
        card_close()
        return

    chart_df = (
        pd.DataFrame(
            {
                "category": [c.name.replace("_", " ") for c in mix.categories],
                "weight_%": [round(c.weight * 100, 1) for c in mix.categories],
            }
        )
        .set_index("category")
        .sort_values("weight_%", ascending=False)
    )
    st.bar_chart(chart_df, height=260)

    rows = [
        {
            "Category": cat.name.replace("_", " "),
            "Share %": round(cat.weight * 100, 1),
            "Why": cat.rationale,
            "Example episode IDs": ", ".join(cat.representative_episode_ids[:3]),
        }
        for cat in sorted(mix.categories, key=lambda c: c.weight, reverse=True)
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    card_close()


def render_gaps(mix: TrainingMix) -> None:
    if not mix.gap_analysis:
        card_open("Gaps in your data", "lime", "skills the index did not cover")
        st.success(
            "Good news: every skill we flagged for this task showed up in at least one retrieved episode. "
            "You may still want new demos for robustness."
        )
        card_close()
        return

    card_open(
        "Gaps in your data",
        "magenta",
        f"{len(mix.gap_analysis)} skill(s) your indexed demos probably don’t teach well enough",
    )
    st.warning(
        "The rows below are **not** generic advice — they’re tied to *this* task. "
        "Plan new recordings (or curate more old data) for those skills."
    )
    for g in mix.gap_analysis:
        skill_label = g.skill.replace("_", " ")
        html(
            f'<div class="bsm-gap-card">'
            f'<div class="bsm-gap-title">{skill_label} · priority {g.importance:.0%}</div>'
            f'<p><strong>Why it matters:</strong> {g.why_it_matters}</p>'
            f'<p><strong>If you skip it:</strong> {g.what_fails_without_it}</p>'
            f'<p class="bsm-gap-fix"><strong>Smallest fix:</strong> {g.minimal_fix}</p>'
            f"</div>"
        )
    card_close()


def render_config(mix: TrainingMix) -> None:
    cfg = mix.training_config
    card_open(
        "Settings file for your trainer",
        "brass",
        "JSON you can adapt for BC / imitation / your stack",
    )
    if cfg is None:
        st.info("No structured config this run.")
        card_close()
        return

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Sampling**")
        st.caption(cfg.sampling_strategy)
        st.markdown("**Curriculum (order)**")
        if cfg.curriculum:
            tag_row(cfg.curriculum, color="brass")
        else:
            st.caption("Single stage — no ordering hint.")
    with cols[1]:
        st.markdown("**Loss multipliers** (higher = train harder on that bucket)")
        lw_df = pd.DataFrame(
            {
                "Bucket": [k.replace("_", " ") for k in cfg.loss_weighting.keys()],
                "Multiplier": [round(v, 3) for v in cfg.loss_weighting.values()],
            }
        ).sort_values("Multiplier", ascending=False)
        st.dataframe(lw_df, use_container_width=True, hide_index=True)

    st.markdown("**Notes from the recommender**")
    st.caption(cfg.notes)

    config_json = json.dumps(cfg.model_dump(), indent=2)
    with st.expander("View full JSON (advanced)", expanded=False):
        st.code(config_json, language="json")
    st.download_button(
        label="Download training_config.json",
        data=config_json,
        file_name="training_config.json",
        mime="application/json",
        use_container_width=True,
    )
    card_close()


def render_explanation(bundle: AnalysisBundle) -> None:
    card_open("Full write-up", "cyan", "training strategy + how the task was read")
    st.markdown(bundle.training_mix.explanation)
    with st.expander("Technical: decomposition notes"):
        st.markdown(bundle.analysis.explanation or "_No extra notes._")
    card_close()


def render_results(
    bundle: AnalysisBundle, dataset_source: str, indexed_count: int
) -> Callable[[], None]:
    """Render tabbed results + status chips."""

    def _go() -> None:
        llm = bundle.llm
        if llm.used_llm:
            llm_chip = status_chip("AI", f"{llm.model} ✓", "ok")
        elif llm.fallback_reason == "user_disabled_llm":
            llm_chip = status_chip("AI", "rules only", "neutral")
        else:
            llm_chip = status_chip("AI", "fallback", "warn")

        dataset_kind = (
            "alert" if dataset_source == "droid" else ("ok" if dataset_source == "aist" else "neutral")
        )
        gap_kind = "alert" if bundle.training_mix.gap_analysis else "ok"
        chips = [
            status_chip("Library", f"{indexed_count:,} eps", dataset_kind),
            status_chip("Retrieved", f"{len(bundle.retrieved)}", "ok"),
            status_chip("Gaps", str(len(bundle.training_mix.gap_analysis)), gap_kind),
            llm_chip,
        ]
        html('<div class="bsm-stripe">' + "".join(chips) + "</div>")

        html('<h2 class="bsm-results-heading">Your analysis</h2>')
        st.caption("Explore by tab — everything updates together for this task.")

        t_sum, t_skill, t_ep, t_plan, t_out = st.tabs(
            [
                "Summary",
                "Skills & risks",
                "Similar episodes",
                "Training plan",
                "Export & notes",
            ]
        )

        with t_sum:
            render_overview(bundle, dataset_source, indexed_count)

        with t_skill:
            render_briefing(bundle.analysis)

        with t_ep:
            render_salvage(bundle.retrieved)

        with t_plan:
            render_mix(bundle.training_mix)
            render_gaps(bundle.training_mix)

        with t_out:
            render_config(bundle.training_mix)
            render_explanation(bundle)

    return _go
