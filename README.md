# BiSkill Miner

**Bimanual skill retrieval and post-training data curation — in one flow.**

---

## In simple words

You describe a **new two-arm robot task** in English (for example: *“crack an egg into a pan and add salt”*). BiSkill Miner:

1. **Figures out what skills** the task needs (grasping, stabilizing, pouring, sequencing, and so on).
2. **Lists ways a trained policy could fail** on that task — not vague “it might fail,” but concrete failure patterns.
3. **Searches your indexed demonstration library** (mock data, AIST task families, or a small DROID text subset) for episodes that might help.
4. **Says what’s missing** if your library cannot cover a skill, and suggests the **smallest amount of new data** that would fix it.
5. **Outputs a structured training recipe** (mix of data, curriculum hint, loss weights) as JSON you can adapt for imitation learning or fine-tuning.

**What this is not:** it does not train a robot policy or run the real robot. It is a **planning and curation tool** that sits *before* training: *which old episodes matter, what gaps remain, and how to mix data sensibly.*

---

## Research framing (why this exists)

### Problem

Post-training and imitation learning for **bimanual** manipulation rarely fail for lack of *gigabytes* of generic data. They fail because:

- **Transfer is selective.** Old episodes that look “similar” by object name may not teach the coordination or contact skills the new task needs.
- **Coverage is sparse in skill space.** A new task may require skills (e.g. delicate grasp, sprinkling, stabilize-then-act) that existing logs underrepresent.
- **Failure modes are predictable.** Policies tend to fail in recurring ways (timing between arms, slip, over-force, sequencing). Surfacing those *before* training improves evaluation design and data targeting.

BiSkill Miner treats curation as a **skill- and coordination-aware retrieval problem** with explicit **gap analysis** and a **machine-readable training configuration**, instead of ad hoc “use more data.”

### Method (high level)

1. **Task analysis** — Map natural language to a fixed **skill taxonomy** and **bimanual coordination taxonomy** (see `src/taxonomy.py`). An optional **large language model** (Qwen via Ollama) proposes structured outputs with retries and validation; a **rule-based analyzer** guarantees a usable path when the LLM is off or unavailable. Outputs include **predicted failure modes** (LLM plus taxonomy floor).
2. **Retrieval** — Episodes are embedded with a sentence encoder (`all-MiniLM-L6-v2` by default); **FAISS** (inner product on L2-normalized vectors ≈ cosine similarity) returns candidates. A **skill-aware retriever** adjusts scoring, reports **per-episode overlap** with required skills, coordination match, and a **task-specific usefulness** string (`src/retriever.py`).
3. **Recommendation** — A **training mix** allocates weight across buckets (target demos vs. skill-aligned retrieved groups). **Gap analysis** lists required skills with no adequate retrieved support, with **why it matters**, **what fails without it**, and a **minimal_fix** prescription (`src/recommender.py`, grounded in `SKILL_GAP_KNOWLEDGE` in `taxonomy.py`). When gaps exist, the mix can favor **target-task demonstrations** and emit a **curriculum-oriented** `TrainingConfig`.

### Outputs (what you can cite or log)

- **`TaskAnalysis`** — skills, weights, coordination, failure modes, explanation.
- **`RetrievedEpisode`** — similarity, matched skills, overlap %, coordination flag, structured `match_reason`.
- **`TrainingMix`** — mix, categories, `gap_analysis`, narrative `explanation`.
- **`LLMRunInfo`** — whether the LLM ran, model id, attempts, latency, fallback reason (no silent degradation).
- **`TrainingConfig`** — JSON-serializable knobs: `dataset_mix`, `sampling_strategy`, `loss_weighting`, `per_category_episodes`, `curriculum`, `notes`.

---

## Architecture

```
Natural-language task
         │
         ▼
┌────────────────────┐
│ Task analysis      │  src/llm.py — Ollama (Qwen) + few-shot + retries
│                    │           — rule fallback + LLMRunInfo telemetry
│                    │           — failure_modes (taxonomy-augmented)
└────────────────────┘
         │ TaskAnalysis
         ▼
┌────────────────────┐
│ Embedding + FAISS  │  src/embeddings.py, src/indexer.py
│ retrieval          │  src/retriever.py — skill/coord bonuses, explanations
└────────────────────┘
         │ list[RetrievedEpisode]
         ▼
┌────────────────────┐
│ Mix + gaps + cfg │  src/recommender.py — TrainingMix, TrainingConfig
└────────────────────┘
         │
         ▼
   Streamlit UI      app.py + src/ui_sections.py — tabbed summary, skills,
                      similar episodes, training plan, export
```

---

## Project layout

```
BiSkill Miner/
├── README.md
├── requirements.txt
├── app.py                    # Streamlit entry: landing, workflow, sidebar
├── static/
│   ├── theme.css             # Themed layout, tabs, accessibility helpers
│   └── hero.png / hero.gif   # Optional hero art (you can add hero.gif)
├── .streamlit/config.toml    # Dark theme, optional static serving
├── data/
│   ├── mock_episodes.json
│   ├── aist_episodes.json    # Built from AIST task table (optional)
│   ├── droid_episodes.json   # Built from tiny DROID subset (optional)
│   └── index/                # Per-dataset FAISS + metadata (generated)
├── scripts/
│   ├── generate_mock_episodes.py
│   ├── parse_aist_task_list.py
│   ├── build_aist_dataset.py
│   ├── build_droid_subset.py # HF or TFDS backend
│   ├── smoke_test.py
│   └── smoke_test_droid_json.py
└── src/
    ├── __init__.py           # Thread env guards (PyTorch / FAISS stability)
    ├── config.py             # DATASET_SOURCE: mock | aist | droid
    ├── taxonomy.py           # Skills, coordination, failure + gap KB
    ├── schemas.py
    ├── llm.py
    ├── embeddings.py
    ├── indexer.py
    ├── retriever.py
    ├── recommender.py
    ├── droid_loader.py       # HF + TFDS; LeRobot meta for instructions
    ├── ui_sections.py        # Tabbed results rendering
    ├── loaders/aist.py
    └── utils.py
```

---

## Quickstart

```bash
cd "BiSkill Miner"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: stronger task decomposition via local LLM
# ollama pull qwen2.5:7b-instruct

streamlit run app.py
```

**Default dataset** is `mock` (`DATASET_SOURCE` in `src/config.py`). The first run may download the embedding model (~80 MB) and build a FAISS index under `data/index/`.

**Offline embeddings:** point Hugging Face cache to the project if needed:

```bash
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_CACHE="$PWD/.hf_cache"
```

---

## Datasets (choose one)

| Source | What it is | How to use |
|--------|--------------|------------|
| **`mock`** | In-repo synthetic episodes | Works out of the box |
| **`aist`** | [AIST Bimanual Manipulation](https://aistairc.github.io/aist_bimanip_site/) task families as episodes | Run `python scripts/build_aist_dataset.py` after `parse_aist_task_list.py` |
| **`droid`** | Tiny **text-only** slice of DROID-style demos (not 1.7 TB) | Build JSON first — see below |

```bash
export DATASET_SOURCE=mock    # or aist or droid
streamlit run app.py
```

If `DATASET_SOURCE=droid` and `data/droid_episodes.json` is missing, the app **errors with build instructions** (no silent fallback to mock).

### DROID subset (metadata only)

**Recommended — HuggingFace `datasets`** (works on Python 3.14; no TensorFlow):

```bash
pip install datasets
python scripts/build_droid_subset.py --backend hf --max-episodes 100
DATASET_SOURCE=droid streamlit run app.py
```

Uses [`lerobot/droid_100`](https://huggingface.co/datasets/lerobot/droid_100) streaming and reads **LeRobot `meta/episodes.jsonl`** so natural-language instructions are preserved.

**Alternative — TensorFlow Datasets** (Python ≤3.13 where TensorFlow wheels exist):

```bash
pip install tensorflow tensorflow-datasets
python scripts/build_droid_subset.py --backend tfds --split "train[:0.01%]" --max-episodes 100
```

Default `--backend auto` tries HF first, then TFDS.

---

## Schema reference (compact)

| Artifact | Role |
|----------|------|
| `TaskAnalysis` | `required_skills`, `skill_weights`, `required_coordination`, `failure_modes`, `explanation` |
| `RetrievedEpisode` | episode + `similarity_score`, `matched_skills`, `skill_overlap_pct`, `coord_overlap`, `match_reason` |
| `TrainingMix` | `recommended_mix`, `categories`, `gap_analysis`, `training_config`, `explanation` |
| `TrainingConfig` | `dataset_mix`, `sampling_strategy`, `loss_weighting`, `per_category_episodes`, `curriculum`, `notes` |
| `LLMRunInfo` | `used_llm`, `model`, `attempts`, `duration_ms`, `fallback_reason`, `last_error` |

Full definitions: `src/schemas.py`.

---

## Example behavior (egg task, mock library)

Task: *“crack an egg into a pan and add salt.”*

- **Skills** may include sequencing, delicate grasp, container interaction, tool use, sprinkling, etc.
- **Coordination** often emphasizes asymmetric or stabilize-then-act patterns.
- **Retrieval** surfaces episodes whose *skills* overlap even when object names differ.
- **Gap analysis** may flag skills no retrieved episode covers well (e.g. **sprinkling**) with a **minimal_fix** (e.g. a small set of shaker / dispense demos).
- **`training_config.json`** may up-weight **target_task_demos** and suggest a **curriculum** when gaps exist.

Exact numbers depend on the active corpus and whether the LLM path is on.

---

## LLM integration (research note)

Prompting in `src/llm.py` constrains outputs to the **taxonomy**, requests **failure_modes**, uses **few-shot** JSON examples, and validates / filters unknown labels. **Ollama** JSON mode, bounded retries, and **LLMRunInfo** ensure users see when analysis is rule-based versus model-based — important for reproducibility and ablations.

---

## Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASET_SOURCE` | `mock` | `mock`, `aist`, or `droid` |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | LLM endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model name |
| `OLLAMA_TIMEOUT_SEC` | `60` | Per-attempt timeout |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Encoder |
| `EMBED_DEVICE` | `cpu` | Use `cuda` when available |
| `TOP_K_DEFAULT` | `20` | Retrieval depth (UI can override) |
| `USE_LLM` | `true` | Disable via `false` for reproducible rule-only runs |

---

## Testing before you push

```bash
python -m py_compile app.py src/*.py src/loaders/*.py scripts/*.py
python scripts/smoke_test.py
python scripts/smoke_test_droid_json.py   # no DROID download; validates droid JSON path
```

---

## Limitations (honest scope)

- **Text-centric retrieval** — no video or force-tape in the default pipeline; episodes are labeled and embedded as text + taxonomy fields.
- **Taxonomy ceiling** — analysis quality is bounded by `taxonomy.py`; new behaviors need new tags or richer descriptions.
- **Corpus scale** — DROID integration is intentional **metadata / instruction** subsetting for laptops, not full multi-terabyte ingestion.
- **Policy-agnostic** — outputs advise **data mixing and curriculum**; your trainer still owns the algorithm (BC, diffusion policy, etc.).

---

## Roadmap (brief)

- Stronger embeddings / optional cross-encoder reranking.
- Vision-language episode features (optional fusion).
- Native export profiles (e.g. LeRobot YAML) on top of `TrainingConfig`.
- Richer gap → **collection checklist** (scene, objects, success criteria).

---

## Design principles

1. **Taxonomy is the contract** — one place (`taxonomy.py`) for skills, coordination, failure priors, and gap copy.
2. **No silent LLM fallback** — every run exposes **LLMRunInfo** in the UI.
3. **Actionable gaps** — not “collect more data,” but **what fails** and **minimal fix** language.
4. **One JSON artifact** — `training_config.json` for tooling and version control.
5. **Swap the corpus, keep the pipeline** — same code path for mock, AIST, or DROID-derived episodes.

---

## Citation and data sources

If you use real corpora in publications, cite the **original datasets** (e.g. AIST Bimanual Manipulation, DROID) and describe BiSkill Miner as a **curation / retrieval layer** over their (or your own) exported episode lists. This repository’s default mock data is for development only.

---

## License

Ensure compliance with **third-party dataset licenses** (AIST, DROID, Hugging Face mirrors) when you build `*_episodes.json` files and share them.
