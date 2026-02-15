#!/usr/bin/env python3
"""
Kernel Bench Progress Reporter.

Generates progress summary and detailed markdown reports for kernel bench sessions.

Usage:
    python kb_score.py <session_id>
    python kb_score.py <session_id> <task_name>
    python kb_score.py                          # Uses most recent session
    python kb_score.py --chain <chain_id>       # Chain cross-batch report
    python kb_score.py <chain_id>               # Auto-detect chain
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Completion reasons that indicate a task should be retryable on resume
RETRYABLE_REASONS = ("server_error", "all_iterations_failed")

# Safety-net fallback for max_iterations when session manifest is missing or
# corrupted. The authoritative default lives in the skill command (kernel-bench.md,
# --iterations=20). This value is deliberately conservative (10) and visually
# distinct so that sessions running on the fallback are obvious in reports.
_FALLBACK_MAX_ITERATIONS = 10


def _is_task_retryable(result: dict) -> bool:
    """Check if a completed task should be treated as retryable on resume."""
    reason = result.get("completion_reason", "")
    if reason in RETRYABLE_REASONS:
        return True
    # Legacy fallback: no completion_reason but 0 speedup with "failed" strategy
    if result.get("speedup", 0) == 0 and result.get("strategy") == "failed" and not reason:
        return True
    return False


def get_output_base():
    """Get the base output directory for sessions."""
    return Path(os.path.expanduser("~/.inference/claude_code_output"))


def get_most_recent_session():
    """Get the most recently modified session ID."""
    base = get_output_base()
    if not base.exists():
        return None
    sessions = [d for d in base.iterdir() if d.is_dir() and (d / "session_manifest.json").exists()]
    if not sessions:
        return None
    # Sort by modification time
    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return sessions[0].name


def load_session_data(session_id: str) -> dict:
    """Load all session data including manifest and task progress."""
    base = get_output_base()
    session_dir = base / session_id
    manifest_file = session_dir / "session_manifest.json"

    if not manifest_file.exists():
        return {"error": f"Session not found: {session_id}"}

    try:
        manifest = json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return {"error": f"Corrupted manifest for session: {session_id}"}

    tasks_data = []
    all_tasks = manifest.get("tasks", [])

    for task_name in all_tasks:
        task_dir = session_dir / task_name
        task_info = {
            "name": task_name,
            "status": "pending",
            "worker": None,
            "iterations": [],
            "best_speedup": None,
            "best_iteration": None,
            "best_strategy": None,
            "completed": False,
            "has_correct_result": False,
            "completion_reason": None
        }

        if not task_dir.exists():
            tasks_data.append(task_info)
            continue

        # Load progress.json
        progress_file = task_dir / "progress.json"
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text())
                task_info["iterations"] = progress.get("iterations", [])
                task_info["worker"] = progress.get("worker_id")

                best = progress.get("best")
                if best:
                    task_info["best_speedup"] = best.get("speedup")
                    task_info["best_iteration"] = best.get("iteration")
                    task_info["best_strategy"] = best.get("strategy")

                # Check if any iteration is correct
                for it in task_info["iterations"]:
                    if it.get("compiled") and it.get("correct"):
                        task_info["has_correct_result"] = True
                        break
            except json.JSONDecodeError:
                pass

        # Check if completed
        if (task_dir / "best_result.json").exists():
            try:
                best_result = json.loads((task_dir / "best_result.json").read_text())
                if _is_task_retryable(best_result):
                    task_info["status"] = "retryable"
                    task_info["completion_reason"] = best_result.get("completion_reason", "unknown")
                else:
                    task_info["status"] = "completed"
                    task_info["completed"] = True
                    task_info["completion_reason"] = best_result.get("completion_reason")
            except (json.JSONDecodeError, Exception):
                task_info["status"] = "completed"
                task_info["completed"] = True
        elif (task_dir / ".in_progress").exists():
            task_info["status"] = "in_progress"
        elif task_info["iterations"]:
            task_info["status"] = "incomplete"

        tasks_data.append(task_info)

    return {
        "session_id": session_id,
        "level": manifest.get("level"),
        "config": manifest.get("config", {}),
        "created_at": manifest.get("created_at"),
        "tasks": tasks_data
    }


def compute_summary(session_data: dict) -> dict:
    """Compute summary statistics for the session."""
    tasks = session_data.get("tasks", [])
    total = len(tasks)
    max_iterations = session_data.get("config", {}).get("max_iterations", _FALLBACK_MAX_ITERATIONS)

    if total == 0:
        return {
            "total": 0,
            "completed": 0,
            "in_progress": 0,
            "pending": 0,
            "failed": 0,
            "correctness_ratio": 0.0,
            "fast_1_3_ratio": 0.0,
            "avg_speedup": 0.0
        }

    completed = sum(1 for t in tasks if t["status"] == "completed")
    in_progress = sum(1 for t in tasks if t["status"] == "in_progress")
    incomplete = sum(1 for t in tasks if t["status"] == "incomplete")
    pending = sum(1 for t in tasks if t["status"] == "pending")
    retryable = sum(1 for t in tasks if t["status"] == "retryable")

    # Count tasks with correct results (compiled & correct)
    correct_count = sum(1 for t in tasks if t["has_correct_result"])

    # Count tasks with speedup >= 1.3
    fast_1_3_count = sum(1 for t in tasks if t["best_speedup"] and t["best_speedup"] >= 1.3)

    # Calculate average speedup (only for tasks with valid speedup)
    speedups = [t["best_speedup"] for t in tasks if t["best_speedup"] and t["best_speedup"] > 0]
    avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

    # Failed = all iterations failed OR incomplete with max_iterations+ failed iterations
    failed = 0
    for t in tasks:
        if t["iterations"] and not t["has_correct_result"]:
            if len(t["iterations"]) >= max_iterations:
                failed += 1

    return {
        "total": total,
        "completed": completed,
        "in_progress": in_progress,
        "pending": pending + incomplete,
        "retryable": retryable,
        "failed": failed,
        "correctness_ratio": round(correct_count / total, 3) if total > 0 else 0.0,
        "fast_1_3_ratio": round(fast_1_3_count / total, 3) if total > 0 else 0.0,
        "fast_1_3_count": fast_1_3_count,
        "correct_count": correct_count,
        "avg_speedup": round(avg_speedup, 3)
    }


def generate_markdown_report(session_data: dict, summary: dict) -> str:
    """Generate a detailed markdown report."""
    session_id = session_data["session_id"]
    level = session_data.get("level", "unknown")
    config = session_data.get("config", {})
    tasks = session_data.get("tasks", [])
    max_iterations = config.get("max_iterations", _FALLBACK_MAX_ITERATIONS)

    lines = []

    # Header
    lines.append(f"# Kernel Bench Progress Report")
    lines.append("")
    lines.append(f"**Session:** `{session_id}`")
    lines.append(f"**Level:** {level}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Key Metrics Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tasks | {summary['total']} |")
    lines.append(f"| Completed | {summary['completed']} |")
    lines.append(f"| In Progress | {summary['in_progress']} |")
    lines.append(f"| Pending | {summary['pending']} |")
    lines.append(f"| Retryable | {summary['retryable']} |")
    lines.append(f"| Failed | {summary['failed']} |")
    lines.append(f"| **Correctness Ratio** | {summary['correctness_ratio']:.1%} ({summary['correct_count']}/{summary['total']}) |")
    lines.append(f"| **Fast (≥1.3x)** | {summary['fast_1_3_ratio']:.1%} ({summary['fast_1_3_count']}/{summary['total']}) |")
    lines.append(f"| Avg Speedup | {summary['avg_speedup']:.3f}x |")
    lines.append("")

    # Config info if available
    if config:
        lines.append("### Configuration")
        lines.append("")
        if config.get("original_command"):
            lines.append(f"**Command:** `{config['original_command']}`")
        lines.append(f"- Workers: {config.get('num_workers', 'N/A')}")
        lines.append(f"- Provider: {config.get('provider', 'N/A')}")
        lines.append("")

    # Sort tasks for display
    status_order = {"in_progress": 0, "incomplete": 1, "pending": 2, "failed": 3, "completed": 4}
    sorted_tasks = sorted(tasks, key=lambda x: (status_order.get(x["status"], 5), x["name"]))

    # In-Progress Tasks
    in_progress_tasks = [t for t in sorted_tasks if t["status"] == "in_progress"]
    if in_progress_tasks:
        lines.append("## In Progress")
        lines.append("")
        lines.append("| Task | Worker | Iterations | Best Speedup |")
        lines.append("|------|--------|------------|--------------|")
        for t in in_progress_tasks:
            speedup_str = f"{t['best_speedup']:.2f}x" if t["best_speedup"] else "-"
            lines.append(f"| {t['name']} | {t['worker'] or '-'} | {len(t['iterations'])}/{max_iterations} | {speedup_str} |")
        lines.append("")

    # Completed Tasks (sorted by speedup)
    completed_tasks = [t for t in sorted_tasks if t["status"] == "completed"]
    if completed_tasks:
        completed_tasks.sort(key=lambda x: x["best_speedup"] or 0, reverse=True)
        lines.append("## Completed Tasks")
        lines.append("")
        lines.append("| Task | Speedup | Strategy | Iterations |")
        lines.append("|------|---------|----------|------------|")
        for t in completed_tasks:
            speedup_str = f"{t['best_speedup']:.2f}x" if t["best_speedup"] else "-"
            strategy = t["best_strategy"] or "-"
            if len(strategy) > 30:
                strategy = strategy[:27] + "..."
            lines.append(f"| {t['name']} | {speedup_str} | {strategy} | {len(t['iterations'])} |")
        lines.append("")

    # Retryable Tasks (server errors, all-failed with 0x)
    retryable_tasks = [t for t in sorted_tasks if t["status"] == "retryable"]
    if retryable_tasks:
        lines.append("## Retryable Tasks")
        lines.append("")
        lines.append("These tasks failed due to server errors or all iterations failing. They will be retried on resume.")
        lines.append("")
        lines.append("| Task | Reason | Iterations | Last Error |")
        lines.append("|------|--------|------------|------------|")
        for t in retryable_tasks:
            reason = t.get("completion_reason", "unknown")
            last_error = "-"
            if t["iterations"]:
                last_it = t["iterations"][-1]
                if last_it.get("error"):
                    last_error = last_it["error"][:50] + "..." if len(last_it.get("error", "")) > 50 else last_it["error"]
            lines.append(f"| {t['name']} | {reason} | {len(t['iterations'])} | {last_error} |")
        lines.append("")

    # Failed Tasks
    failed_tasks = [t for t in sorted_tasks if t["status"] == "incomplete" and not t["has_correct_result"] and len(t["iterations"]) >= max_iterations]
    if failed_tasks:
        lines.append("## Failed Tasks")
        lines.append("")
        lines.append("| Task | Iterations | Last Error |")
        lines.append("|------|------------|------------|")
        for t in failed_tasks:
            last_error = "-"
            if t["iterations"]:
                last_it = t["iterations"][-1]
                if last_it.get("error"):
                    last_error = last_it["error"][:50] + "..." if len(last_it.get("error", "")) > 50 else last_it["error"]
            lines.append(f"| {t['name']} | {len(t['iterations'])} | {last_error} |")
        lines.append("")

    # Pending Tasks
    pending_tasks = [t for t in sorted_tasks if t["status"] == "pending"]
    if pending_tasks and len(pending_tasks) <= 20:
        lines.append("## Pending Tasks")
        lines.append("")
        for t in pending_tasks:
            lines.append(f"- {t['name']}")
        lines.append("")
    elif pending_tasks:
        lines.append(f"## Pending Tasks ({len(pending_tasks)} remaining)")
        lines.append("")
        # Show first 10
        for t in pending_tasks[:10]:
            lines.append(f"- {t['name']}")
        lines.append(f"- ... and {len(pending_tasks) - 10} more")
        lines.append("")

    return "\n".join(lines)


def print_console_summary(session_id: str, summary: dict):
    """Print a quick console summary."""
    print(f"\n   === Session: {session_id} ===\n")
    print(f"     Total:         {summary['total']}")
    print(f"     Completed:     {summary['completed']}")
    print(f"     In Progress:   {summary['in_progress']}")
    print(f"     Pending:       {summary['pending']}")
    print(f"     Retryable:     {summary['retryable']}")
    print(f"     Failed:        {summary['failed']}")
    print(f"     Avg Speedup:   {summary['avg_speedup']:.3f}x")
    print()
    print(f"   --- Key Metrics ---")
    print(f"     Correctness:   {summary['correctness_ratio']:.1%} ({summary['correct_count']}/{summary['total']})")
    print(f"     Fast (>=1.3x): {summary['fast_1_3_ratio']:.1%} ({summary['fast_1_3_count']}/{summary['total']})")
    print()


# ─── Chain Reporting ───


def load_chain_manifest(chain_id):
    """Load chain_manifest.json for a chain."""
    base = get_output_base()
    chain_dir = base / chain_id
    manifest_file = chain_dir / "chain_manifest.json"
    if not manifest_file.exists():
        return None
    try:
        return json.loads(manifest_file.read_text())
    except json.JSONDecodeError:
        return None


def compute_chain_summary(manifest):
    """Compute cross-batch stats from chain manifest and session data."""
    batches = manifest.get("batches", [])
    base = get_output_base()

    # Compute cumulative best per task across all batches
    all_best = {}
    batch_summaries = []

    for b in batches:
        sid = b.get("session_id", "")
        session_data = load_session_data(sid)
        if "error" in session_data:
            batch_summaries.append({
                "batch_index": b.get("batch_index", 0),
                "level": b.get("level", "?"),
                "task_count": b.get("task_count", 0),
                "passed": 0,
                "rate": 0,
                "status": b.get("status", "?"),
            })
            continue

        batch_summary = compute_summary(session_data)

        # Update cumulative best
        for t in session_data.get("tasks", []):
            name = t["name"]
            speedup = t.get("best_speedup") or 0
            if name not in all_best or speedup > all_best[name]:
                all_best[name] = speedup

        cumulative_total = len(all_best)
        cumulative_passing = sum(1 for s in all_best.values() if s >= 1.3)
        cumulative_rate = cumulative_passing / cumulative_total if cumulative_total > 0 else 0

        batch_summaries.append({
            "batch_index": b.get("batch_index", 0),
            "level": b.get("level", "?"),
            "task_count": b.get("task_count", 0),
            "passed": batch_summary.get("fast_1_3_count", 0),
            "rate": batch_summary.get("fast_1_3_ratio", 0),
            "avg_speedup": batch_summary.get("avg_speedup", 0),
            "cumulative_total": cumulative_total,
            "cumulative_passing": cumulative_passing,
            "cumulative_rate": cumulative_rate,
            "status": b.get("status", "?"),
        })

    return batch_summaries, all_best


def print_chain_summary(chain_id, manifest, batch_summaries):
    """Print cross-batch improvement table."""
    print(f"\n   === Chain: {chain_id} ===\n")

    print("   Batch  Level    Tasks  Passed  Rate     Cumulative  Delta")
    print("   -----  ------   -----  ------  ------   ----------  ------")

    prev_cumulative = 0
    for bs in batch_summaries:
        cum_rate = bs.get("cumulative_rate", 0)
        cum_passing = bs.get("cumulative_passing", 0)
        cum_total = bs.get("cumulative_total", 0)

        if bs["batch_index"] == 0:
            delta_str = "  --"
        else:
            delta_pp = (cum_rate - prev_cumulative) * 100
            delta_str = f"+{delta_pp:.1f}pp" if delta_pp >= 0 else f"{delta_pp:.1f}pp"

        status_marker = "" if bs["status"] == "completed" else f" [{bs['status']}]"

        print(f"   {bs['batch_index']:5d}  {bs['level']:<7s}  {bs['task_count']:5d}  {bs['passed']:6d}  "
              f"{bs['rate']:5.1%}    "
              f"{cum_passing}/{cum_total} ({cum_rate:.1%})  {delta_str}{status_marker}")

        prev_cumulative = cum_rate

    print()

    # Final cumulative stats
    cumulative = manifest.get("cumulative", {})
    if cumulative:
        print(f"   Overall: {cumulative.get('tasks_passing', 0)}/{cumulative.get('tasks_total', 0)} "
              f"({cumulative.get('cumulative_success_rate', 0):.1%}), "
              f"avg speedup {cumulative.get('avg_speedup', 0):.3f}x")
        print()


def find_improved_tasks(manifest, all_best):
    """Find tasks that improved across batches."""
    base = get_output_base()
    batches = manifest.get("batches", [])
    improvements = []

    # For each task that appears in multiple batches, find improvement
    task_history = {}  # task_name -> [(batch_index, speedup)]
    for b in batches:
        sid = b.get("session_id", "")
        session_data = load_session_data(sid)
        if "error" in session_data:
            continue
        for t in session_data.get("tasks", []):
            name = t["name"]
            speedup = t.get("best_speedup") or 0
            if name not in task_history:
                task_history[name] = []
            task_history[name].append((b.get("batch_index", 0), speedup))

    for name, history in task_history.items():
        if len(history) < 2:
            continue
        first_speedup = history[0][1]
        last_speedup = history[-1][1]
        if last_speedup > first_speedup and last_speedup >= 1.3 and first_speedup < 1.3:
            improvements.append({
                "name": name,
                "first": first_speedup,
                "final": last_speedup,
                "batches": len(history),
            })

    improvements.sort(key=lambda x: x["final"] - x["first"], reverse=True)
    return improvements


def generate_chain_report(chain_id, manifest, batch_summaries, all_best):
    """Generate chain progress markdown report."""
    lines = []
    lines.append("# Chain Progress Report")
    lines.append("")
    lines.append(f"**Chain:** `{chain_id}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Batches:** {len(batch_summaries)}")
    lines.append("")

    # Cross-batch improvement table
    lines.append("## Cross-Batch Improvement")
    lines.append("")
    lines.append("| Batch | Level | Tasks | Passed | Rate | Cumulative | Delta |")
    lines.append("|-------|-------|-------|--------|------|------------|-------|")

    prev_cumulative = 0
    for bs in batch_summaries:
        cum_rate = bs.get("cumulative_rate", 0)
        cum_passing = bs.get("cumulative_passing", 0)
        cum_total = bs.get("cumulative_total", 0)

        if bs["batch_index"] == 0:
            delta_str = "--"
        else:
            delta_pp = (cum_rate - prev_cumulative) * 100
            delta_str = f"+{delta_pp:.1f}pp" if delta_pp >= 0 else f"{delta_pp:.1f}pp"

        lines.append(f"| {bs['batch_index']} | {bs['level']} | {bs['task_count']} | "
                     f"{bs['passed']} | {bs['rate']:.1%} | "
                     f"{cum_passing}/{cum_total} ({cum_rate:.1%}) | {delta_str} |")
        prev_cumulative = cum_rate

    lines.append("")

    # Task improvements
    improvements = find_improved_tasks(manifest, all_best)
    if improvements:
        lines.append("## Tasks That Improved Across Batches")
        lines.append("")
        lines.append("| Task | First Speedup | Final Speedup | Batches |")
        lines.append("|------|---------------|---------------|---------|")
        for imp in improvements[:20]:
            lines.append(f"| {imp['name']} | {imp['first']:.2f}x | {imp['final']:.2f}x | {imp['batches']} |")
        lines.append("")

    # Overall stats
    cumulative = manifest.get("cumulative", {})
    lines.append("## Overall Stats")
    lines.append("")
    lines.append(f"- **Total tasks:** {cumulative.get('tasks_total', 0)}")
    lines.append(f"- **Passing (>=1.3x):** {cumulative.get('tasks_passing', 0)}")
    lines.append(f"- **Success rate:** {cumulative.get('cumulative_success_rate', 0):.1%}")
    lines.append(f"- **Avg speedup:** {cumulative.get('avg_speedup', 0):.3f}x")
    lines.append("")

    return "\n".join(lines)


def main():
    args = sys.argv[1:]

    # Check for --chain flag
    is_chain = "--chain" in args
    if is_chain:
        args = [a for a in args if a != "--chain"]

    # Parse arguments
    if not args:
        session_id = get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
    elif args[0] in ["-h", "--help"]:
        print(__doc__)
        sys.exit(0)
    else:
        session_id = args[0]

    # Auto-detect chain: check for chain_manifest.json
    base = get_output_base()
    chain_dir = base / session_id
    if not is_chain and (chain_dir / "chain_manifest.json").exists():
        is_chain = True

    if is_chain:
        # Chain reporting mode
        manifest = load_chain_manifest(session_id)
        if not manifest:
            print(f"Chain not found or corrupted: {session_id}")
            sys.exit(1)

        batch_summaries, all_best = compute_chain_summary(manifest)
        print_chain_summary(session_id, manifest, batch_summaries)

        # Show task improvements
        improvements = find_improved_tasks(manifest, all_best)
        if improvements:
            print(f"   --- Tasks Improved Across Batches ({len(improvements)}) ---")
            for imp in improvements[:10]:
                print(f"     {imp['name']}: {imp['first']:.2f}x -> {imp['final']:.2f}x")
            if len(improvements) > 10:
                print(f"     ... and {len(improvements) - 10} more")
            print()

        # Generate and save report
        report = generate_chain_report(session_id, manifest, batch_summaries, all_best)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = chain_dir / f"chain_progress_{timestamp}.md"
        report_file.write_text(report)
        print(f"   Detailed report: {report_file}")
        print()
        sys.exit(0)

    task_name = args[1] if len(args) > 1 else None

    # Load session data
    session_data = load_session_data(session_id)

    if "error" in session_data:
        print(f"Error: {session_data['error']}")
        sys.exit(1)

    # Single task detail
    if task_name:
        max_iterations = session_data.get("config", {}).get("max_iterations", _FALLBACK_MAX_ITERATIONS)
        task = next((t for t in session_data["tasks"] if t["name"] == task_name), None)
        if not task:
            print(f"Task not found: {task_name}")
            sys.exit(1)

        print(f"\n=== Task: {task_name} ===")
        print(f"Status: {task['status']} | Worker: {task['worker'] or '-'} | Iterations: {len(task['iterations'])}/{max_iterations}")
        print()

        if task["iterations"]:
            print("| Iter | Strategy | Compiled | Correct | Speedup | Runtime | Error |")
            print("|------|----------|----------|---------|---------|---------|-------|")
            for it in task["iterations"]:
                compiled = "✓" if it.get("compiled") else "✗"
                correct = "✓" if it.get("correct") else "✗"
                speedup = f"{it.get('speedup', 0):.2f}x" if it.get("speedup") else "-"
                runtime = f"{it.get('runtime_ms', 0):.3f}ms" if it.get("runtime_ms") else "-"
                error = it.get("error", "-") or "-"
                if len(error) > 20:
                    error = error[:17] + "..."
                print(f"| {it.get('iteration', '-')} | {it.get('strategy', '-')[:20]} | {compiled} | {correct} | {speedup} | {runtime} | {error} |")
            print()

        if task["best_speedup"]:
            print(f"Best: iteration {task['best_iteration']}, {task['best_speedup']:.2f}x ({task['best_strategy']})")
        sys.exit(0)

    # Compute summary
    summary = compute_summary(session_data)

    # Print console summary
    print_console_summary(session_id, summary)

    # Generate markdown report
    report = generate_markdown_report(session_data, summary)

    # Save report
    output_base = get_output_base()
    session_dir = output_base / session_id
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = session_dir / f"progress_{timestamp}.md"
    report_file.write_text(report)

    print(f"   Detailed report: {report_file}")
    print()


if __name__ == "__main__":
    main()
