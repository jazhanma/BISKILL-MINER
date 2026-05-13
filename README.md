# BiSkill Miner

**Bimanual skill retrieval and training-data curation from language.**

---

## Why I built this

I’m **Jaskaran Singh**. I wanted a tool that treats **post-training and imitation learning for two-arm robots** as a **curation problem**, not a “download more hours” problem.

When you move from one bimanual task to another, failure usually isn’t random: **transfer is selective**, **coverage in skill space is uneven**, and **failure modes repeat** (timing between arms, contact slips, too much force, bad sequencing). I built BiSkill Miner so I could:

1. **Turn a new task described in natural language** into **structured skills and coordination needs** grounded in an explicit taxonomy.
2. **Surface concrete failure patterns** before training, so evaluation and data collection stay targeted.
3. **Retrieve relevant past demonstrations** from a real episode index (semantic search + skill-aware scoring), instead of guessing which old logs might help.
4. **Quantify what’s missing** when the library doesn’t cover a skill—and emit a **clear, machine-readable training plan** (mixes, curriculum hints, loss weights) I can plug into an imitation or fine-tuning stack.

This project does **not** train policies or command hardware. It sits **upstream of training**: **what to reuse, what’s missing, and how to mix data deliberately.**

---

## What it does

You describe a **new two-arm task** in English. The pipeline:

1. **Analyzes the task** — maps language to a fixed **skill** and **bimanual coordination** taxonomy (`src/taxonomy.py`). Optionally uses **Qwen via Ollama** with validation and retries; a **rule-based path** still returns a consistent result if the LLM is unavailable.
2. **Retrieves episodes** — sentence embeddings (`all-MiniLM-L6-v2` by default) and **FAISS** similarity, with **skill- and coordination-aware** scoring and short **explanations** per hit (`src/retriever.py`).
3. **Recommends a training mix** — allocates weight across retrieved groups, runs **gap analysis** with actionable **minimal_fix** language (`src/recommender.py`, `SKILL_GAP_KNOWLEDGE` in `taxonomy.py`), and exports **`TrainingConfig`** as JSON.

**Transparency:** every run records **`LLMRunInfo`** so you always know whether analysis used the LLM or the rule-based fallback.

---

## How it works (compact)

| Stage | Implementation |
|--------|----------------|
| Task analysis | `src/llm.py` — Ollama JSON mode, few-shot, taxonomy constraints, failure modes |
| Index | `src/embeddings.py`, `src/indexer.py` — encoder + FAISS + episode metadata |
| Retrieval | `src/retriever.py` — overlap, coordination match, `match_reason` |
| Recommendation | `src/recommender.py` — `TrainingMix`, gaps, `training_config.json` |
| UI | `app.py`, `src/ui_sections.py` — Streamlit workflow, exports |

---

## Architecture

```
Natural-language task
         │
         ▼
┌────────────────────┐
│ Task analysis      │  src/llm.py — Ollama (Qwen) + rule fallback + LLMRunInfo
└────────────────────┘
         │ TaskAnalysis
         ▼
┌────────────────────┐
│ Embedding + FAISS  │  src/embeddings.py, src/indexer.py, src/retriever.py
└────────────────────┘
         │ list[RetrievedEpisode]
         ▼
┌────────────────────┐
│ Mix + gaps + JSON  │  src/recommender.py — TrainingMix, TrainingConfig
└────────────────────┘
         ▼
      Streamlit UI
```

---

## Repository layout

```
BiSkill Miner/
├── README.md
├── requirements.txt
├── app.py
├── static/               # theme.css, optional hero image
├── .streamlit/config.toml
├── data/
│   ├── aist_episodes.json    # built from AIST sources (optional)
│   ├── droid_episodes.json   # built from DROID / LeRobot metadata (optional)
│   └── index/                 # FAISS + metadata (generated)
├── scripts/
│   ├── parse_aist_task_list.py, build_aist_dataset.py
│   ├── build_droid_subset.py   # Hugging Face or TFDS
│   ├── smoke_test.py, smoke_test_droid_json.py
└── src/
    ├── config.py, taxonomy.py, schemas.py
    ├── llm.py, embeddings.py, indexer.py, retriever.py, recommender.py
    ├── droid_loader.py, loaders/aist.py, ui_sections.py, utils.py
```

---

## Quickstart

```bash
cd "BiSkill Miner"
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: richer task decomposition (local)
# ollama pull qwen2.5:7b-instruct

# Index a real corpus first (see Datasets), then:
export DATASET_SOURCE=aist   # or droid
streamlit run app.py
```

First run downloads the embedding model (~80 MB) and builds a FAISS index under `data/index/`.

**Offline / air-gapped caches** (optional):

```bash
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_CACHE="$PWD/.hf_cache"
```

---

## Datasets

BiSkill Miner expects an **episodes JSON** for the corpus you select with **`DATASET_SOURCE`**.

| `DATASET_SOURCE` | Corpus | Notes |
|------------------|--------|--------|
| **`aist`** | [AIST Bimanual Manipulation](https://aistairc.github.io/aist_bimanip_site/) | Task families exported to `data/aist_episodes.json` via the scripts in `scripts/`. |
| **`droid`** | DROID-style **metadata** (instructions + labels), not the full recording corpus | Build a laptop-sized JSON from LeRobot / Hugging Face or TFDS (see below). |

```bash
export DATASET_SOURCE=aist   # or droid
streamlit run app.py
```

### DROID subset (recommended path on modern Python)

Use **Hugging Face `datasets`** (no TensorFlow required):

```bash
pip install datasets
python scripts/build_droid_subset.py --backend hf --max-episodes 100
DATASET_SOURCE=droid streamlit run app.py
```

Uses [`lerobot/droid_100`](https://huggingface.co/datasets/lerobot/droid_100) and **LeRobot `meta/episodes.jsonl`** so natural-language instructions are preserved.

**Alternative — TensorFlow Datasets** (where TensorFlow wheels exist for your Python version):

```bash
pip install tensorflow tensorflow-datasets
python scripts/build_droid_subset.py --backend tfds --split "train[:0.01%]" --max-episodes 100
```

`--backend auto` tries Hugging Face first, then TFDS.

### AIST

Prepare the task list, then run `python scripts/build_aist_dataset.py` as documented in `scripts/`. Point `DATASET_SOURCE=aist` at the generated `data/aist_episodes.json`.

---

## Main artifacts

| Name | Role |
|------|------|
| `TaskAnalysis` | Required skills, weights, coordination, failure modes, explanation |
| `RetrievedEpisode` | Scores, overlaps, coordination flags, `match_reason` |
| `TrainingMix` | Recommended mix, categories, gap analysis, narrative |
| `TrainingConfig` | `dataset_mix`, sampling, loss weighting, curriculum, `notes` |
| `LLMRunInfo` | Whether the LLM ran, model id, timing, fallback reason |

Full types: `src/schemas.py`.

---

## Configuration

| Variable | Default | Meaning |
|----------|---------|---------|
| `DATASET_SOURCE` | (see `src/config.py`) | Set to **`aist`** or **`droid`** for the workflows above; other values are for internal tooling. |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | LLM endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Model name |
| `OLLAMA_TIMEOUT_SEC` | `60` | Per-attempt timeout |
| `EMBED_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Encoder |
| `EMBED_DEVICE` | `cpu` | Use `cuda` when available |
| `TOP_K_DEFAULT` | `20` | Retrieval depth |
| `USE_LLM` | `true` | Set `false` for reproducible rule-only analysis |

---

## Tests

```bash
python -m py_compile app.py src/*.py src/loaders/*.py scripts/*.py
python scripts/smoke_test.py
python scripts/smoke_test_droid_json.py
```

---

## Limitations (scope)

- **Text-first retrieval** — default pipeline uses language and metadata; video and force traces are not first-class features here.
- **Taxonomy-bound** — new behaviors need taxonomy and description updates in `taxonomy.py`.
- **DROID integration** — intentional **metadata / instruction** subset for practical indexing, not a full multi-terabyte mirror.
- **Policy-agnostic** — outputs guide **data and curriculum**; your trainer still owns the algorithm.

---

## License and data compliance

Respect **third-party dataset licenses** (AIST, DROID, Hugging Face distributions) when you build and share `*_episodes.json` files.

If you use this software in research, cite the **underlying datasets** you indexed and describe BiSkill Miner as a **retrieval and curation layer** on top of your episode exports.
