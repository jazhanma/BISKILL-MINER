"""Parse the AIST Bimanual Manipulation Dataset task list.

Source: https://aistairc.github.io/aist_bimanip_site/
We use the offline copy at uploads/aist_bimanip_site-1.md so this is reproducible
without network access.

Produces ``data/aist_task_list.json``: a clean structured list of all task
families with task_id, task_name (snake_case), aist_taxonomy, skill_verb,
num_episodes, date, and download_url.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(
    "/Users/jaskarangrewal/.cursor/projects/Users-jaskarangrewal-BiSkill-Miner/"
    "uploads/aist_bimanip_site-1.md"
)
OUTPUT_PATH = REPO_ROOT / "data" / "aist_task_list.json"


CELL_DOWNLOAD_RE = re.compile(r"\[Download\]\(([^)]+)\)")


def _clean_cell(text: str) -> str:
    return text.strip().replace("\\_", "_").replace("\\&", "&")


def _parse_row(line: str) -> Optional[Dict]:
    """Parse one markdown table row into a structured dict."""
    line = line.strip()
    if not line.startswith("|") or "Total" in line or "Task ID" in line:
        return None
    cells = [c for c in (cell.strip() for cell in line.strip("|").split("|"))]
    if len(cells) < 8:
        return None
    task_id_raw = _clean_cell(cells[0])
    if not task_id_raw.isdigit():
        return None
    task_name = _clean_cell(cells[1])
    aist_taxonomy = _clean_cell(cells[3])
    skill_verb = _clean_cell(cells[4])
    num_str = _clean_cell(cells[5])
    date = _clean_cell(cells[6])
    download_cell = cells[7]
    m = CELL_DOWNLOAD_RE.search(download_cell)
    download_url = m.group(1) if m else None
    try:
        num_episodes = int(num_str)
    except ValueError:
        num_episodes = 0
    return {
        "task_id": int(task_id_raw),
        "task_name": task_name,
        "aist_taxonomy": aist_taxonomy,
        "skill_verb": skill_verb,
        "num_episodes": num_episodes,
        "date": date,
        "download_url": download_url,
    }


def parse_markdown(md_path: Path) -> List[Dict]:
    text = md_path.read_text(encoding="utf-8")
    rows: List[Dict] = []
    for line in text.splitlines():
        parsed = _parse_row(line)
        if parsed is not None:
            rows.append(parsed)
    rows.sort(key=lambda r: r["task_id"])
    return rows


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not src.exists():
        raise FileNotFoundError(f"AIST source markdown not found: {src}")
    tasks = parse_markdown(src)
    if not tasks:
        raise RuntimeError("No task rows parsed from the source markdown.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    total_ep = sum(t["num_episodes"] for t in tasks)
    print(f"Wrote {len(tasks)} AIST task families ({total_ep:,} episodes) to {OUTPUT_PATH}")

    by_skill: Dict[str, int] = {}
    by_taxo: Dict[str, int] = {}
    for t in tasks:
        by_skill[t["skill_verb"]] = by_skill.get(t["skill_verb"], 0) + 1
        by_taxo[t["aist_taxonomy"]] = by_taxo.get(t["aist_taxonomy"], 0) + 1
    print(f"  unique skill verbs:  {len(by_skill)}: {sorted(by_skill.keys())}")
    print(f"  unique AIST taxonomy: {len(by_taxo)}: {sorted(by_taxo.keys())}")


if __name__ == "__main__":
    main()
