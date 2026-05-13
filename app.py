"""BiSkill Miner — Streamlit UI (friendly workflow + themed console).

Run:  streamlit run app.py
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List

import streamlit as st

from src.config import (
    DATASET_IS_METADATA_ONLY,
    DATASET_LABEL,
    DATASET_SOURCE,
    DATASET_URL,
    EPISODES_PATH,
    FAISS_INDEX_PATH,
    OLLAMA_MODEL,
    TOP_K_DEFAULT,
    USE_LLM_DEFAULT,
)
from src.indexer import EpisodeIndex, get_or_build_index
from src.llm import analyze_task, check_ollama
from src.recommender import recommend_training_mix
from src.retriever import retrieve_relevant_episodes
from src.schemas import AnalysisBundle, LLMRunInfo
from src.ui_sections import html, render_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("biskill.app")

PROJECT_ROOT = Path(__file__).resolve().parent
THEME_CSS = PROJECT_ROOT / "static" / "theme.css"
HERO_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "static" / "hero.gif",
    PROJECT_ROOT / "static" / "hero.png",
    PROJECT_ROOT / "static" / "hero.jpg",
    PROJECT_ROOT / "static" / "hero.webp",
]

st.set_page_config(
    page_title="BiSkill Miner — bimanual training helper",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXAMPLE_TASKS: List[str] = [
    "crack an egg into a pan and add salt",
    "fold a towel in half",
    "open a jar",
    "pour juice into a cup while holding the cup",
    "wipe a table while holding an object steady",
    "assemble two parts together",
    "stir batter in a bowl",
    "pack three items into a box",
]

QUICK_EXAMPLES = [
    ("Egg + salt", "crack an egg into a pan and add salt"),
    ("Fold towel", "fold a towel in half"),
    ("Open jar", "open a jar"),
    ("Pour & hold", "pour juice into a cup while holding the cup"),
]


def _apply_theme() -> None:
    try:
        css = THEME_CSS.read_text(encoding="utf-8")
    except FileNotFoundError:
        log.warning("Theme CSS not found at %s", THEME_CSS)
        return
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading your episode library…")
def _load_index() -> EpisodeIndex:
    return get_or_build_index()


@st.cache_data(show_spinner=False)
def _ollama_status_cached(model: str) -> LLMRunInfo:
    return check_ollama(model=model)


def _run_pipeline(task: str, use_llm: bool, top_k: int, idx: EpisodeIndex) -> AnalysisBundle:
    analysis, run_info = analyze_task(task, use_llm=use_llm)
    retrieved = retrieve_relevant_episodes(analysis, top_k=top_k, index=idx)
    mix = recommend_training_mix(task, retrieved, analysis)
    return AnalysisBundle(
        task=task,
        analysis=analysis,
        retrieved=retrieved,
        training_mix=mix,
        llm=run_info,
    )


@st.cache_data(show_spinner=False)
def _hero_image_data_uri() -> str | None:
    for candidate in HERO_IMAGE_CANDIDATES:
        if candidate.exists():
            mime = {
                ".gif": "image/gif",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }.get(candidate.suffix.lower(), "image/png")
            b64 = base64.b64encode(candidate.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
    return None


def _hero() -> None:
    img_uri = _hero_image_data_uri()
    media_block = (
        f"""
        <div class="bsm-media">
            <span class="bsm-bracket bsm-bracket-tl"></span>
            <span class="bsm-bracket bsm-bracket-tr"></span>
            <span class="bsm-bracket bsm-bracket-bl"></span>
            <span class="bsm-bracket bsm-bracket-br"></span>
            <div class="bsm-media-frame">
                <img src="{img_uri}" alt="BiSkill Miner visual" />
            </div>
            <div class="bsm-media-meta">
                <span>Visual</span>
                <span>READY</span>
                <span>BIMANUAL</span>
            </div>
        </div>
        """
        if img_uri
        else """
        <div class="bsm-media">
            <div class="bsm-media-frame" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);font-family:Inter,sans-serif;font-size:0.9rem;text-align:center;padding:1.2rem;">
                Add <code>static/hero.png</code> or <code>static/hero.gif</code> for a hero image.
            </div>
        </div>
        """
    )

    html(
        f"""
        <div class="bsm-landing">
            <div>
                <div class="bsm-eyebrow">Plan better robot fine-tuning · less guesswork</div>
                <h1>BiSkill <span class="bsm-accent">Miner</span></h1>
                <div class="bsm-subtitle">Turn a task description into skills, matching demos, and a training recipe.</div>
                <p class="bsm-tagline">
                    Describe what you want a <strong>bimanual</strong> robot to do. This tool breaks the task into
                    skills, flags what might go wrong, pulls similar episodes from <em>your</em> indexed dataset,
                    and suggests how to mix data for fine-tuning — including what’s still missing from your library.
                </p>
                <div class="bsm-pillars">
                    <span class="bsm-pillar"><span class="bsm-pillar-num">1</span> Understand the task</span>
                    <span class="bsm-pillar"><span class="bsm-pillar-num">2</span> Spot risks</span>
                    <span class="bsm-pillar"><span class="bsm-pillar-num">3</span> Find useful old demos</span>
                    <span class="bsm-pillar"><span class="bsm-pillar-num">4</span> Get a training mix + JSON</span>
                </div>
                <div class="bsm-prompt-strip">
                    <span class="bsm-prompt-cursor">▍</span>
                    <span>Step 1: type a task below, then press <strong>Run analysis</strong></span>
                </div>
            </div>
            {media_block}
        </div>
        """
    )


def _workflow_steps() -> None:
    html(
        """
        <div class="bsm-workflow">
            <div class="bsm-workflow-title">How it works</div>
            <ol class="bsm-workflow-list">
                <li><strong>Describe</strong> the new task in everyday language.</li>
                <li><strong>Run analysis</strong> — takes a few seconds on a laptop.</li>
                <li>Open the <strong>Summary</strong> tab first, then drill into skills, episodes, and training plan.</li>
                <li><strong>Download</strong> <code>training_config.json</code> from the Export tab when you’re happy.</li>
            </ol>
        </div>
        """
    )


def _render_sidebar(idx_holder: dict) -> tuple[bool, int]:
    with st.sidebar:
        html(
            '<div class="bsm-sidebar-brand">'
            '<div class="bsm-sidebar-eyebrow">Controls</div>'
            '<div class="bsm-sidebar-title">BiSkill Miner</div></div>'
        )

        st.markdown("### AI analysis")
        use_llm = st.toggle(
            "Use local AI (Ollama / Qwen)",
            value=USE_LLM_DEFAULT,
            help="When on, uses your Ollama model if it’s running. When off, fast rule-based analysis only.",
        )
        top_k = st.slider(
            "How many similar episodes to show",
            min_value=5,
            max_value=40,
            value=TOP_K_DEFAULT,
            step=1,
            help="Larger = more rows in the Similar episodes tab. 15–25 is usually enough.",
        )

        st.markdown("### Data source")
        st.markdown(f"Active: **`{DATASET_SOURCE}`**")
        st.caption(DATASET_LABEL)
        if DATASET_URL:
            st.markdown(f"[Learn about this dataset]({DATASET_URL})")
        if DATASET_SOURCE == "droid":
            st.warning("Using a small DROID **metadata** slice only — not the full multi‑TB release.")
        elif DATASET_SOURCE == "mock":
            st.caption("Demo data built into the repo — good for trying the UI.")
        st.caption("Switch with an environment variable before starting: `DATASET_SOURCE=mock`, `aist`, or `droid`.")

        st.markdown("### Library status")
        try:
            idx = _load_index()
            idx_holder["idx"] = idx
            st.success(f"Ready — **{len(idx.episodes):,}** episodes indexed")
            st.caption(f"File: `{EPISODES_PATH.name}`")
            st.caption(f"Vector index: `{FAISS_INDEX_PATH.name}`")
            if DATASET_IS_METADATA_ONLY:
                st.caption("Indexed text + labels only (no video files loaded).")
        except Exception as exc:
            st.error(f"Could not load episodes: {exc}")
            st.stop()

        st.markdown("### AI server")
        status = _ollama_status_cached(OLLAMA_MODEL)
        if status.used_llm:
            st.success(f"Ollama reachable — `{status.model}`")
        else:
            st.info(
                f"Ollama not in use ({status.fallback_reason or 'unavailable'}). "
                "The app still works with built-in rules."
            )

        with st.expander("More sample tasks", expanded=False):
            for ex in EXAMPLE_TASKS:
                if st.button(ex[:56] + ("…" if len(ex) > 56 else ""), key=f"ex_{ex}", use_container_width=True):
                    st.session_state["task_input"] = ex

    return use_llm, top_k


def main() -> None:
    if "task_input" not in st.session_state:
        st.session_state["task_input"] = EXAMPLE_TASKS[0]

    _apply_theme()

    idx_holder: dict = {}
    use_llm, top_k = _render_sidebar(idx_holder)
    idx = idx_holder["idx"]

    _hero()
    _workflow_steps()

    st.markdown("### Your task")
    task = st.text_input(
        "Describe what the robot should do (both arms if relevant)",
        value=st.session_state.get("task_input", EXAMPLE_TASKS[0]),
        placeholder='Example: "pick up the bowl with one arm and stir with the other"',
        label_visibility="visible",
        help="Be specific about objects and both arms. You can edit anytime and run again.",
    )

    st.caption("Quick fill:")
    qc = st.columns(len(QUICK_EXAMPLES))
    for col, (label, full_text) in zip(qc, QUICK_EXAMPLES):
        with col:
            if st.button(label, key=f"quick_{label}", use_container_width=True):
                st.session_state["task_input"] = full_text
                st.rerun()

    btn_row = st.columns([1, 2])
    with btn_row[0]:
        run = st.button("Run analysis", type="primary", use_container_width=True)
    with btn_row[1]:
        st.caption(
            "One click runs: task understanding → episode search → training mix + optional gaps + downloadable config."
        )

    if not run:
        return
    if not task.strip():
        st.error("Add a short description of the task, then try again.")
        return

    st.session_state["task_input"] = task.strip()

    with st.spinner("Analyzing task, searching your library, building recommendations…"):
        bundle = _run_pipeline(task.strip(), use_llm=use_llm, top_k=top_k, idx=idx)

    st.markdown("---")
    render_results(bundle, DATASET_SOURCE, len(idx.episodes))()

    st.caption("Change the task or settings in the sidebar and run again to compare.")


if __name__ == "__main__":
    main()
