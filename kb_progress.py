#!/usr/bin/env python3
"""Kernel Bench Progress Reporter.

Usage:
    python3 kb_progress.py {session_id}           # Quick summary + generate detailed report
    python3 kb_progress.py {session_id} --json    # Raw JSON output
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def get_batch_progress(session_id: str) -> tuple[dict, Path]:
    """Read session data and return (data, session_dir)."""
    output_base = Path.home() / ".inference" / "claude_code_output"
    session_dir = output_base / session_id
    manifest_file = session_dir / "session_manifest.json"

    if not manifest_file.exists():
        print(f"Session '{session_id}' not found at {session_dir}")
        sys.exit(1)

    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        print(f"Corrupted manifest for session: {session_id}")
        sys.exit(1)

    all_tasks = manifest.get("tasks", [])

    tasks_progress = []
    completed_count = 0
    in_progress_count = 0
    failed_count = 0
    pending_count = 0
    all_speedups = []

    for task_name in all_tasks:
        task_dir = session_dir / task_name
        progress_file = task_dir / "progress.json"

        task_info = {
            "name": task_name,
            "status": "pending",
            "worker": None,
            "iterations_planned": 3,
            "iterations_done": 0,
            "best_speedup": None,
            "last_iteration": None,
            "iterations": []
        }

        if not task_dir.exists():
            pending_count += 1
            tasks_progress.append(task_info)
            continue

        progress = {}
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
            except json.JSONDecodeError:
                pass

        iterations = progress.get("iterations", [])
        task_info["worker"] = progress.get("worker_id")
        task_info["iterations_planned"] = progress.get("config", {}).get("max_iterations", 3)
        task_info["iterations_done"] = len(iterations)
        task_info["iterations"] = iterations

        if iterations:
            task_info["last_iteration"] = {
                "compiled": iterations[-1].get("compiled"),
                "correct": iterations[-1].get("correct"),
                "speedup": iterations[-1].get("speedup"),
                "error": iterations[-1].get("error")
            }

        best = progress.get("best")
        if best:
            task_info["best_speedup"] = best.get("speedup")

        best_result_file = task_dir / "best_result.json"
        if best_result_file.exists():
            task_info["status"] = "completed"
            completed_count += 1
            if not task_info["best_speedup"]:
                try:
                    best_result = json.loads(best_result_file.read_text())
                    task_info["best_speedup"] = best_result.get("speedup")
                except:
                    pass
            if task_info["best_speedup"]:
                all_speedups.append(task_info["best_speedup"])
        else:
            marker_file = task_dir / ".in_progress"
            if marker_file.exists():
                try:
                    marker_data = json.loads(marker_file.read_text())
                    task_info["status"] = "in_progress"
                    task_info["worker"] = marker_data.get("worker_id") or marker_data.get("worker")
                    in_progress_count += 1
                except json.JSONDecodeError:
                    task_info["status"] = "incomplete"
                    pending_count += 1
            elif iterations:
                if all(not it.get("correct") for it in iterations) and len(iterations) >= 3:
                    task_info["status"] = "failed"
                    failed_count += 1
                else:
                    task_info["status"] = "incomplete"
                    pending_count += 1
            else:
                pending_count += 1

        tasks_progress.append(task_info)

    avg_speedup = sum(all_speedups) / len(all_speedups) if all_speedups else 0

    data = {
        "summary": {
            "total": len(all_tasks),
            "completed": completed_count,
            "in_progress": in_progress_count,
            "failed": failed_count,
            "pending": pending_count,
            "avg_speedup": round(avg_speedup, 3)
        },
        "tasks": tasks_progress
    }

    return data, session_dir


def format_iter_result(it: dict) -> str:
    """Format a single iteration result."""
    compiled = it.get("compiled", False)
    correct = it.get("correct", False)
    speedup = it.get("speedup", 0)

    if compiled and correct:
        return f"{speedup:.2f}x ✓"
    elif compiled:
        return "incorrect ✗"
    else:
        return "compile ✗"


def print_summary(data: dict, session_id: str, report_path: Path):
    """Print quick summary to stdout."""
    s = data["summary"]

    print(f"═══ Session: {session_id} ═══")
    print()
    print(f"  Total:       {s['total']}")
    print(f"  Completed:   {s['completed']}")
    print(f"  In Progress: {s['in_progress']}")
    print(f"  Pending:     {s['pending']}")
    print(f"  Failed:      {s['failed']}")
    print(f"  Avg Speedup: {s['avg_speedup']}x")
    print()
    print(f"📄 Detailed report: {report_path}")


def generate_markdown_report(data: dict, session_id: str, session_dir: Path) -> Path:
    """Generate detailed markdown report and save to session directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = session_dir / f"progress_{timestamp}.md"

    s = data["summary"]
    lines = []

    # Header
    lines.append(f"## Session: {session_id}")
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Summary table
    lines.append("### Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| **Total** | {s['total']} |")
    lines.append(f"| **Completed** | {s['completed']} |")
    lines.append(f"| **In Progress** | {s['in_progress']} |")
    lines.append(f"| **Pending** | {s['pending']} |")
    lines.append(f"| **Failed** | {s['failed']} |")
    lines.append(f"| **Avg Speedup** | {s['avg_speedup']}x |")
    lines.append("")

    # In-progress tasks
    in_progress = [t for t in data["tasks"] if t["status"] == "in_progress"]
    if in_progress:
        lines.append(f"### In-Progress ({len(in_progress)})")
        lines.append("")
        lines.append("| Task | Worker | Iter | Status |")
        lines.append("|------|--------|------|--------|")
        for t in in_progress:
            name = t["name"]
            worker = t.get("worker") or "-"
            iters_done = t.get("iterations_done", 0)
            iters_planned = t.get("iterations_planned", 3)
            last = t.get("last_iteration", {})
            if last:
                if last.get("compiled") and last.get("correct"):
                    status = f"✓ {last.get('speedup', 0)}x"
                elif last.get("compiled"):
                    status = "✗ incorrect"
                else:
                    status = "✗ compile error"
            else:
                status = "-"
            lines.append(f"| {name} | {worker} | {iters_done}/{iters_planned} | {status} |")
        lines.append("")

    # Completed tasks
    completed = [t for t in data["tasks"] if t["status"] == "completed"]
    completed = sorted(completed, key=lambda x: x.get("best_speedup", 0) or 0, reverse=True)

    if completed:
        lines.append(f"### All Completed ({len(completed)}) - Sorted by Speedup")
        lines.append("")
        lines.append("| Task | Best | Iteration Path |")
        lines.append("|------|------|----------------|")
        for t in completed:
            name = t["name"]
            best = t.get("best_speedup", 0) or 0
            iters = t.get("iterations", [])
            if iters:
                iter_parts = [format_iter_result(it) for it in iters]
                iter_str = "  →  ".join(iter_parts)
            else:
                iter_str = "-"
            lines.append(f"| {name} | **{best:.2f}x** | {iter_str} |")
        lines.append("")

    # Failed tasks
    failed = [t for t in data["tasks"] if t["status"] == "failed"]
    if failed:
        lines.append(f"### Failed ({len(failed)})")
        lines.append("")
        for t in failed:
            name = t["name"]
            iters = t.get("iterations", [])
            if iters:
                iter_parts = [format_iter_result(it) for it in iters]
                iter_str = "  →  ".join(iter_parts)
            else:
                iter_str = "-"
            lines.append(f"- **{name}**: {iter_str}")
        lines.append("")

    # Pending tasks
    pending = [t for t in data["tasks"] if t["status"] in ("pending", "incomplete")]
    if pending:
        lines.append(f"### Pending ({len(pending)})")
        lines.append("")
        names = [t["name"] for t in pending]
        for i in range(0, len(names), 5):
            lines.append(", ".join(names[i:i+5]))
        lines.append("")

    # Write to file
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Kernel Bench Progress Reporter")
    parser.add_argument("session_id", help="Session ID to query")
    parser.add_argument("--json", action="store_true", help="Output raw JSON only")
    args = parser.parse_args()

    data, session_dir = get_batch_progress(args.session_id)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        # Generate detailed markdown report
        report_path = generate_markdown_report(data, args.session_id, session_dir)
        # Print quick summary to stdout
        print_summary(data, args.session_id, report_path)


if __name__ == "__main__":
    main()
