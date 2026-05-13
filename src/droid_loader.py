"""DROID dataset ingestion — metadata-only.

DROID (https://droid-dataset.github.io/) ships ~1.7TB of teleoperated robot
demos. For BiSkill Miner we only need the *task instructions* and a few
episode-level metadata fields, because the system is a data-intelligence
retriever, not a policy trainer (yet). So we:

* Load a tiny subset of DROID.
* Read only the natural-language instruction off each episode.
* Skip every image / video / proprioceptive trajectory tensor we can.
* Convert each sample into a taxonomy-validated ``Episode`` via the
  deterministic rule-based analyzer (NO calls to Qwen/Ollama, so this is
  reproducible and fast on a MacBook CPU).

Two backends, picked via the ``backend`` arg of :func:`load_droid_subset`:

* ``"hf"`` — HuggingFace ``datasets`` streaming from the public
  ``lerobot/droid_100`` mirror (101 episodes, ~MB-scale streaming, pure
  Python — works on Python 3.14 and any other version). **Default.**
* ``"tfds"`` — ``tensorflow_datasets`` against a local ``droid`` corpus
  or the public GCS mirror. Requires ``tensorflow`` which currently
  ships wheels only up to Python 3.13.
* ``"auto"`` — try HF first, fall back to TFDS if ``datasets`` isn't
  installed.

Heavy deps (``tensorflow_datasets``, ``datasets``) are imported lazily
inside the corresponding ``_import_*`` helpers, so the rest of
BiSkill Miner has no TF or HF-datasets dependency.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from .llm import rule_based_analyze
from .schemas import Episode
from .taxonomy import COORDINATION_SET

log = logging.getLogger("biskill.droid_loader")


_INSTRUCTION_KEYS: Sequence[str] = (
    "language_instruction",
    "language_instruction_2",
    "language_instruction_3",
    "natural_language_instruction",
    "task_instruction",
    "instruction",
)


def _import_tfds():
    """Import tensorflow_datasets lazily with a friendly install hint."""
    try:
        import tensorflow_datasets as tfds  # type: ignore
        return tfds
    except ImportError as exc:
        raise ImportError(
            "tensorflow-datasets is required for the TFDS DROID backend. "
            "Install with: pip install tensorflow tensorflow-datasets "
            "(note: TensorFlow ships wheels for Python 3.9-3.13 only; on "
            "Python 3.14 use backend='hf' instead.)"
        ) from exc


def _import_hf_datasets():
    """Import HuggingFace ``datasets`` lazily with a friendly install hint."""
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except ImportError as exc:
        raise ImportError(
            "huggingface 'datasets' is required for the HF DROID backend. "
            "Install with: pip install datasets"
        ) from exc


def _decode(value: Any) -> str:
    """Convert a TFDS eager tensor / bytes / str into a clean Python string."""
    if value is None:
        return ""
    if hasattr(value, "numpy"):
        try:
            value = value.numpy()
        except Exception:
            return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if isinstance(value, str):
        return value.strip()
    try:
        flat = list(value)  # numpy 1-D array of objects / bytes
        if flat:
            head = flat[0]
            if hasattr(head, "decode"):
                return head.decode("utf-8", errors="ignore").strip()
            return str(head).strip()
    except Exception:
        pass
    return ""


def _first_step(sample_steps: Any) -> Optional[Any]:
    """Pull just the first element from a TFDS steps dataset.

    DROID stores ``steps`` as a nested ``tf.data.Dataset``. The language
    instruction is constant across steps so one peek is enough.
    """
    try:
        it = iter(sample_steps)
        return next(it)
    except StopIteration:
        return None
    except Exception:
        return None


def extract_task_instruction(sample: Any) -> str:
    """Best-effort extraction of the natural-language task instruction.

    Tries multiple known keys at the episode level, then peeks the first
    step (where DROID actually stores its instructions), then checks
    ``episode_metadata``. Returns ``"unknown task"`` if nothing is found.
    Never raises.
    """
    if not isinstance(sample, dict):
        return "unknown task"

    for key in _INSTRUCTION_KEYS:
        if key in sample:
            text = _decode(sample[key])
            if text:
                return text

    if "steps" in sample:
        step = _first_step(sample["steps"])
        if isinstance(step, dict):
            for key in _INSTRUCTION_KEYS:
                if key in step:
                    text = _decode(step[key])
                    if text:
                        return text
            obs = step.get("observation")
            if isinstance(obs, dict):
                for key in _INSTRUCTION_KEYS:
                    if key in obs:
                        text = _decode(obs[key])
                        if text:
                            return text

    meta = sample.get("episode_metadata") or sample.get("metadata")
    if isinstance(meta, dict):
        for key in _INSTRUCTION_KEYS:
            if key in meta:
                text = _decode(meta[key])
                if text:
                    return text

    return "unknown task"


def _default_coordination(coords: Sequence[str]) -> str:
    """DROID is a single-arm Franka Panda dataset — default accordingly,
    but let the rule-based analyzer override if the instruction implies
    bimanual coordination."""
    for c in coords:
        if c in COORDINATION_SET:
            return c
    return "single_arm"


def _episode_from_instruction(instruction: str, index: int) -> Episode:
    analysis = rule_based_analyze(instruction or "unknown task")
    return Episode(
        episode_id=f"droid_{index:06d}",
        task_name=instruction,
        description=instruction,
        objects=[],
        skills=list(analysis.required_skills),
        coordination_type=_default_coordination(analysis.required_coordination),
        quality_score=0.85,
    )


def convert_droid_sample_to_episode(sample: Any, index: int) -> Episode:
    """Convert one DROID sample into a taxonomy-validated ``Episode``.

    Uses the deterministic rule-based analyzer (no LLM) to label skills
    and coordination from the instruction text. This guarantees the
    output is reproducible and CPU-fast.
    """
    instruction = extract_task_instruction(sample) or "unknown task"
    return _episode_from_instruction(instruction, index)


def _build_decoders(tfds_mod):
    """Best-effort: tell TFDS to skip decoding video / image tensors.

    DROID's image keys live under ``steps.observation.*``. Skipping their
    decoders avoids spending CPU on JPEG decode — the raw bytes are still
    read, but we never touch them. If the feature names ever change,
    decoder construction fails silently and we just don't skip.
    """
    try:
        skip = tfds_mod.decode.SkipDecoding()
        return {
            "steps": {
                "observation": {
                    "exterior_image_1_left": skip,
                    "exterior_image_2_left": skip,
                    "wrist_image_left": skip,
                }
            }
        }
    except Exception:
        return None


def _resolve_output_path(output_path: str) -> Path:
    p = Path(output_path).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parents[1] / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _write_episodes(episodes: List[Episode], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([ep.model_dump() for ep in episodes], f, indent=2)
    log.info("Wrote %d DROID episodes to %s", len(episodes), out_path)


def _load_via_tfds(
    split: str,
    data_dir: str,
    max_episodes: Optional[int],
) -> List[Episode]:
    tfds = _import_tfds()
    expanded = os.path.expanduser(data_dir)

    decoders = _build_decoders(tfds)
    load_kwargs = {
        "split": split,
        "data_dir": expanded,
        "shuffle_files": False,
        "try_gcs": True,
    }
    if decoders is not None:
        load_kwargs["decoders"] = decoders

    log.info("Loading DROID via tfds: split=%s data_dir=%s", split, expanded)
    try:
        ds = tfds.load("droid", **load_kwargs)
    except TypeError:
        load_kwargs.pop("decoders", None)
        ds = tfds.load("droid", **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DROID via tfds.load(split={split!r}, data_dir={expanded!r}). "
            "Make sure the dataset is prepared locally OR that you have network "
            "access for GCS streaming. We never trigger the full 1.7TB download "
            "automatically.\n"
            f"Underlying error: {exc}"
        ) from exc

    episodes: List[Episode] = []
    iterable: Iterable = ds.take(max_episodes) if max_episodes else ds
    for i, sample in enumerate(iterable):
        try:
            episodes.append(convert_droid_sample_to_episode(sample, i + 1))
        except Exception as exc:
            log.warning("Skipping DROID sample %d: %s", i, exc)
            continue
    return episodes


def _hf_download_file(repo_id: str, filename: str) -> Optional[Path]:
    """Best-effort fetch of a single small file from an HF dataset repo."""
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        return None
    try:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
        return Path(p)
    except Exception as exc:
        log.debug("Could not fetch %s from %s: %s", filename, repo_id, exc)
        return None


def _read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except json.JSONDecodeError:
                continue
    return out


def _load_lerobot_episodes_meta(repo_id: str) -> List[dict]:
    """LeRobot v2: ``meta/episodes.jsonl`` lists one record per episode
    with ``episode_index``, ``tasks`` (list of strings), and ``length``."""
    p = _hf_download_file(repo_id, "meta/episodes.jsonl")
    if p is None:
        return []
    return _read_jsonl(p)


def _load_lerobot_tasks_map(repo_id: str) -> dict:
    """LeRobot v2: ``meta/tasks.jsonl`` maps ``task_index`` to ``task`` text."""
    p = _hf_download_file(repo_id, "meta/tasks.jsonl")
    if p is None:
        return {}
    mapping: dict = {}
    for obj in _read_jsonl(p):
        ti = obj.get("task_index")
        text = obj.get("task") or obj.get("instruction") or obj.get("description")
        if ti is not None and text:
            try:
                mapping[int(ti)] = str(text).strip()
            except (TypeError, ValueError):
                continue
    return mapping


def _instruction_from_episode_meta(ep_meta: dict, tasks_map: dict) -> str:
    """Resolve the per-episode instruction from a LeRobot meta record."""
    tasks = ep_meta.get("tasks")
    if isinstance(tasks, list) and tasks:
        head = tasks[0]
        if isinstance(head, str) and head.strip():
            return head.strip()
        if isinstance(head, int) and head in tasks_map:
            return tasks_map[head]
    for key in ("task", "instruction", "description", "language_instruction"):
        v = ep_meta.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    ti = ep_meta.get("task_index")
    if isinstance(ti, int) and ti in tasks_map:
        return tasks_map[ti]
    return ""


def _load_via_hf_metadata(repo_id: str, max_episodes: Optional[int]) -> List[Episode]:
    """Fast path: read ``meta/episodes.jsonl`` directly. No parquet IO.

    Returns ``[]`` if the metadata file isn't present (caller falls back
    to streaming).
    """
    ep_meta_list = _load_lerobot_episodes_meta(repo_id)
    if not ep_meta_list:
        return []
    tasks_map = _load_lerobot_tasks_map(repo_id)

    ep_meta_list = sorted(
        ep_meta_list,
        key=lambda m: (m.get("episode_index") if isinstance(m.get("episode_index"), int) else 0),
    )

    episodes: List[Episode] = []
    for ep in ep_meta_list:
        ep_idx_raw = ep.get("episode_index")
        try:
            ep_idx = int(ep_idx_raw) if ep_idx_raw is not None else len(episodes)
        except (TypeError, ValueError):
            ep_idx = len(episodes)
        instruction = _instruction_from_episode_meta(ep, tasks_map) or "unknown task"
        try:
            episodes.append(_episode_from_instruction(instruction, ep_idx))
        except Exception as exc:
            log.warning("Skipping HF DROID episode %d: %s", ep_idx, exc)
            continue
        if max_episodes is not None and len(episodes) >= max_episodes:
            break
    return episodes


def _load_via_hf_streaming(
    repo_id: str,
    split: str,
    max_episodes: Optional[int],
) -> List[Episode]:
    """Fallback path: stream the parquet rows.

    LeRobot v2 stores step rows with ``task_index`` (int) — we resolve it
    against ``meta/tasks.jsonl``. Older mirrors may store
    ``language_instruction`` (string) inline; ``extract_task_instruction``
    handles both.
    """
    load_dataset = _import_hf_datasets()
    hf_split = _normalize_hf_split(split)
    tasks_map = _load_lerobot_tasks_map(repo_id)

    log.info(
        "Streaming HuggingFace dataset: repo=%s split=%s", repo_id, hf_split
    )
    try:
        ds = load_dataset(repo_id, split=hf_split, streaming=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to stream HuggingFace dataset {repo_id!r} (split={hf_split!r}). "
            "Verify the repo id and that you have network access to huggingface.co. "
            f"Underlying error: {exc}"
        ) from exc

    episodes: List[Episode] = []
    seen_eps: set = set()
    fallback_idx = 0
    for row in ds:
        if not isinstance(row, dict):
            continue
        ep_idx_raw = row.get("episode_index")
        if ep_idx_raw is None:
            ep_idx_raw = row.get("episode_id")
        if ep_idx_raw is None:
            ep_idx = fallback_idx
            fallback_idx += 1
        else:
            try:
                ep_idx = int(ep_idx_raw)
            except (TypeError, ValueError):
                ep_idx = fallback_idx
                fallback_idx += 1
        if ep_idx in seen_eps:
            continue
        seen_eps.add(ep_idx)

        instruction = extract_task_instruction(row)
        if not instruction and "task_index" in row:
            try:
                ti = int(row["task_index"])
                instruction = tasks_map.get(ti, "")
            except (TypeError, ValueError):
                pass
        if not instruction:
            instruction = "unknown task"

        try:
            episodes.append(_episode_from_instruction(instruction, ep_idx))
        except Exception as exc:
            log.warning("Skipping HF DROID episode %d: %s", ep_idx, exc)
            continue

        if max_episodes is not None and len(episodes) >= max_episodes:
            break

    return episodes


def _load_via_hf(
    repo_id: str,
    split: str,
    max_episodes: Optional[int],
) -> List[Episode]:
    """Load DROID metadata from a HuggingFace mirror.

    Default repo is ``lerobot/droid_100`` (101 episodes). LeRobot v2 stores
    the task text in ``meta/episodes.jsonl`` and a task-index mapping in
    ``meta/tasks.jsonl``, so we read those tiny JSONL files directly
    (~ms-scale, no parquet streaming). If those files are absent we
    fall back to streaming the parquet rows and resolving instructions
    against the tasks map.
    """
    episodes = _load_via_hf_metadata(repo_id=repo_id, max_episodes=max_episodes)
    if episodes:
        log.info(
            "Loaded %d DROID episodes via LeRobot meta/episodes.jsonl (fast path).",
            len(episodes),
        )
        return episodes

    log.info("LeRobot meta files not found — falling back to parquet streaming.")
    return _load_via_hf_streaming(
        repo_id=repo_id, split=split, max_episodes=max_episodes
    )


def _normalize_hf_split(split: str) -> str:
    """Best-effort TFDS-style → HF-style split conversion.

    TFDS ``train[:0.1%]`` -> HF ``train`` (we cap via ``max_episodes`` instead,
    because percentage slicing on streamed HF datasets is unreliable).
    HF ``train[:N]`` is passed through.
    """
    if "%" in split:
        return split.split("[", 1)[0]
    return split


HF_DEFAULT_REPO = "lerobot/droid_100"


def load_droid_subset(
    split: str = "train[:0.1%]",
    data_dir: str = "~/datasets/droid",
    max_episodes: Optional[int] = 500,
    output_path: str = "data/droid_episodes.json",
    backend: str = "auto",
    hf_repo: str = HF_DEFAULT_REPO,
) -> List[Episode]:
    """Load a tiny DROID subset, convert to ``Episode`` records, save JSON.

    Parameters
    ----------
    split
        For TFDS: a slice spec like ``train[:0.1%]``.
        For HF: the base split name (percent-slices are stripped — use
        ``max_episodes`` to cap).
    data_dir
        TFDS data_dir. Ignored by the HF backend.
    max_episodes
        Hard cap on episode count regardless of split size. Default 500.
    output_path
        Where to write the JSON list of ``Episode`` records.
    backend
        ``"hf"`` (default-preferred), ``"tfds"``, or ``"auto"`` (HF first,
        TFDS if ``datasets`` is missing).
    hf_repo
        HF dataset id when ``backend != "tfds"``. Default
        ``lerobot/droid_100`` (101 episodes, ~MB streaming).
    """
    backend = (backend or "auto").lower()
    if backend not in {"auto", "hf", "tfds"}:
        raise ValueError(f"Unknown backend={backend!r}. Use 'hf', 'tfds', or 'auto'.")

    if backend == "auto":
        try:
            _import_hf_datasets()
            backend = "hf"
        except ImportError:
            log.info("HF 'datasets' not installed; falling back to TFDS backend.")
            backend = "tfds"

    if backend == "hf":
        episodes = _load_via_hf(repo_id=hf_repo, split=split, max_episodes=max_episodes)
    else:
        episodes = _load_via_tfds(split=split, data_dir=data_dir, max_episodes=max_episodes)

    out_path = _resolve_output_path(output_path)
    _write_episodes(episodes, out_path)
    return episodes
