from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


CHECK_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s*\[(?P<mark>[xX ~])\]\s+(?P<text>.+?)\s*$")
STATUS_RE = re.compile(r"Статус:\s*`(?P<status>TODO|IN_PROGRESS|DONE|BLOCKED)`", re.IGNORECASE)


def _pct(done: int, total: int) -> float:
    if total <= 0:
        return 100.0
    return (100.0 * float(done)) / float(total)


def analyze_plan(path: Path) -> Dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()

    done = 0
    todo = 0
    in_progress = 0
    raw_items: List[Dict[str, object]] = []

    for idx, line in enumerate(lines, start=1):
        m = CHECK_RE.match(line)
        if m:
            mark = str(m.group("mark")).strip().lower()
            text = str(m.group("text")).strip()
            state = "todo"
            if mark == "x":
                state = "done"
                done += 1
            elif mark == "~":
                state = "in_progress"
                in_progress += 1
            else:
                todo += 1
            raw_items.append({"line": idx, "state": state, "text": text})

    status_done = len([1 for line in lines if "Статус:" in line and "`DONE`" in line])
    status_todo = len([1 for line in lines if "Статус:" in line and "`TODO`" in line])
    status_in_progress = len([1 for line in lines if "Статус:" in line and "`IN_PROGRESS`" in line])
    status_blocked = len([1 for line in lines if "Статус:" in line and "`BLOCKED`" in line])

    total = done + todo + in_progress
    completion_pct = _pct(done, total)
    hard_ready = bool(todo == 0 and in_progress == 0)

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(path),
        "checkboxes": {
            "total": total,
            "done": done,
            "todo": todo,
            "in_progress": in_progress,
            "completion_pct": round(completion_pct, 2),
            "hard_ready": hard_ready,
        },
        "status_lines": {
            "done": status_done,
            "todo": status_todo,
            "in_progress": status_in_progress,
            "blocked": status_blocked,
        },
        "open_items": [row for row in raw_items if str(row.get("state")) in {"todo", "in_progress"}],
    }


def _render_md(payload: Dict[str, object]) -> str:
    c = dict(payload.get("checkboxes", {}))
    s = dict(payload.get("status_lines", {}))
    open_items = list(payload.get("open_items", []))

    lines: List[str] = []
    lines.append("# Plan Completion Report")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- plan_path: `{payload.get('plan_path', '')}`")
    lines.append(f"- completion_pct: `{c.get('completion_pct', 0.0):.2f}%`")
    lines.append(f"- hard_ready: `{c.get('hard_ready', False)}`")
    lines.append(f"- checkboxes_total: `{c.get('total', 0)}`")
    lines.append(f"- checkboxes_done: `{c.get('done', 0)}`")
    lines.append(f"- checkboxes_in_progress: `{c.get('in_progress', 0)}`")
    lines.append(f"- checkboxes_todo: `{c.get('todo', 0)}`")
    lines.append("")
    lines.append("## Status Lines")
    lines.append(f"- DONE: `{s.get('done', 0)}`")
    lines.append(f"- IN_PROGRESS: `{s.get('in_progress', 0)}`")
    lines.append(f"- TODO: `{s.get('todo', 0)}`")
    lines.append(f"- BLOCKED: `{s.get('blocked', 0)}`")
    lines.append("")
    lines.append("## Open Items")
    if not open_items:
        lines.append("- none")
    else:
        for item in open_items:
            lines.append(f"- line {item.get('line', '?')}: [{item.get('state', '')}] {item.get('text', '')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report markdown plan completion from checkbox statuses.")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--strict-threshold", type=float, default=-1.0, help="Fail if completion_pct is below threshold (0..100).")
    args = parser.parse_args()

    plan = Path(str(args.plan)).expanduser().resolve()
    if not plan.exists():
        raise FileNotFoundError(plan)

    payload = analyze_plan(plan)
    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (plan.parent / f"{plan.stem}_PROGRESS.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (plan.parent / f"{plan.stem}_PROGRESS.md")
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"completion_pct: {dict(payload.get('checkboxes', {})).get('completion_pct', 0.0):.2f}%")

    threshold = float(args.strict_threshold)
    if threshold >= 0.0:
        completion = float(dict(payload.get("checkboxes", {})).get("completion_pct", 0.0))
        if completion < threshold:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
