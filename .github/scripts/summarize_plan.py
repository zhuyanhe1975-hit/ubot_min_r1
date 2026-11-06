#!/usr/bin/env python3
"""Write a short GitHub Actions job summary for an example plan run.

Reads the planner log JSON and console log. Appends markdown to
$GITHUB_STEP_SUMMARY when that file is set, and always writes the same
markdown to PLAN_SUMMARY_MD (default: ci-out/summary.md) when that directory
exists or can be created.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


def format_action(action: dict) -> str:
    kind = action.get("kind")
    if kind == "DETACH":
        return f"DETACH({action.get('u')},{action.get('v')})"
    if kind == "ATTACH":
        base = f"ATTACH({action.get('u')},{action.get('pu')},{action.get('v')},{action.get('pv')})"
        twist = action.get("twist")
        if twist is None:
            return base
        return f"ATTACH({action.get('u')},{action.get('pu')},{action.get('v')},{action.get('pv')},twist={twist})"
    if kind == "SET_JOINTS":
        return f"SET_JOINTS({action.get('u')},{action.get('j1')},{action.get('j2')})"
    return str(action)


def _runtime(meta: dict) -> str:
    runtime = meta.get("runtime_sec")
    if isinstance(runtime, (int, float)):
        return f"{float(runtime):.3f} s"
    return "n/a"


def build_summary(exit_code: str, log_path: Path, stdout_path: Path) -> str:
    success = exit_code == "0"
    if exit_code == "":
        result = "failure (planner step did not report an exit code)"
    elif success:
        result = "success"
    else:
        result = f"failure (exit {exit_code})"

    lines = [
        "## UBot example plan",
        "",
        f"**Result:** {result}",
        "",
        "| | |",
        "| --- | --- |",
        "| Start | `examples/start.json` |",
        "| Goal | `examples/goal.json` |",
        "| Command | `python -m ubot.apps.run_plan --start examples/start.json --goal examples/goal.json --planner hier` |",
    ]

    if log_path.is_file():
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            lines.append(f"| Plan log | invalid JSON ({exc}) |")
            data = None
        if isinstance(data, dict):
            meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
            plan = data.get("plan") if isinstance(data.get("plan"), list) else []
            executed = data.get("executed") if isinstance(data.get("executed"), list) else []
            kinds = Counter(
                a.get("kind", "?") for a in plan if isinstance(a, dict)
            )
            kind_text = ", ".join(f"{name} × {count}" for name, count in sorted(kinds.items())) or "n/a"
            lines.extend(
                [
                    f"| Planner | `{meta.get('planner', 'n/a')}` |",
                    f"| Runtime | {_runtime(meta)} |",
                    f"| Actions | {len(plan)} |",
                    f"| Executed | {len(executed)} |",
                    f"| Action kinds | {kind_text} |",
                ]
            )
            lines.append("")
            if plan:
                lines.append("### Actions")
                lines.append("")
                for index, action in enumerate(plan):
                    if isinstance(action, dict):
                        rendered = format_action(action)
                    else:
                        rendered = str(action)
                    lines.append(f"{index:03d}. `{rendered}`")
                lines.append("")
        elif data is not None:
            lines.append("| Plan log | JSON value was not an object |")
            lines.append("")
    else:
        lines.append("| Plan log | not written |")
        lines.append("")

    if stdout_path.is_file():
        text = stdout_path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            tail = "\n".join(text.splitlines()[-40:])
            lines.extend(
                [
                    "<details>",
                    "<summary>Planner output</summary>",
                    "",
                    "```",
                    tail,
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    exit_code = os.environ.get("PLAN_EXIT_CODE", "")
    log_path = Path(os.environ.get("PLAN_LOG", "ci-out/ubot_run_log.json"))
    stdout_path = Path(os.environ.get("PLAN_STDOUT", "ci-out/plan.log"))
    summary = build_summary(exit_code, log_path, stdout_path)

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(summary)

    md_path = Path(os.environ.get("PLAN_SUMMARY_MD", "ci-out/summary.md"))
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(summary, encoding="utf-8")

    if not summary_file:
        print(summary, end="")


if __name__ == "__main__":
    main()
