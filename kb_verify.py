#!/usr/bin/env python3
"""
Kernel Bench Algorithm Verification.

Checks that optimizer agents follow the Phase A/B/C protocol by examining
breadcrumbs (strategy names in progress.json), file artifacts, and content.

Usage:
    python kb_verify.py <session_id>                # All tasks
    python kb_verify.py <session_id> <task_name>    # Single task detail
    python kb_verify.py                             # Most recent session
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime

# Strategy name must start with one of these phase prefixes
PHASE_PATTERN = re.compile(r'^(explore|exploit|revert|switch|algebraic)_')

# Generic names that indicate lazy naming (checked as whole strategy name or suffix)
GENERIC_NAMES = {"triton", "v1", "v2", "v3", "kernel", "attempt", "test", "cuda"}

# Completion reasons that indicate a task was not truly optimized
SKIP_REASONS = ("server_error", "all_iterations_failed")


def get_output_base():
    return Path(os.path.expanduser("~/.inference/claude_code_output"))


def get_most_recent_session():
    base = get_output_base()
    if not base.exists():
        return None
    sessions = [d for d in base.iterdir() if d.is_dir() and (d / "session_manifest.json").exists()]
    if not sessions:
        return None
    sessions.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return sessions[0].name


# ─── Individual Check Functions ───


def check_file_exists(path, name, critical=True):
    """Check that a file exists."""
    return {
        "id": f"file_{name}",
        "name": f"{name} exists",
        "passed": path.exists(),
        "severity": "FAIL" if critical else "WARN",
        "detail": "found" if path.exists() else "missing",
    }


def check_file_nonempty(path, name, critical=True):
    """Check that a file exists and is non-empty."""
    if not path.exists():
        return {
            "id": f"nonempty_{name}",
            "name": f"{name} exists and non-empty",
            "passed": False,
            "severity": "FAIL" if critical else "WARN",
            "detail": "missing",
        }
    content = path.read_text().strip()
    return {
        "id": f"nonempty_{name}",
        "name": f"{name} exists and non-empty",
        "passed": len(content) > 0,
        "severity": "FAIL" if critical else "WARN",
        "detail": f"{len(content)} chars" if content else "empty",
    }


def check_explore_diversity(strategies):
    """Check 1: At least 2 distinct explore_N strategy prefixes."""
    explore_names = set()
    for s in strategies:
        m = re.match(r'explore_\d+_(.+)', s)
        if m:
            explore_names.add(m.group(1))
    return {
        "id": "explore_diversity",
        "name": ">=2 distinct explore strategies",
        "passed": len(explore_names) >= 2,
        "severity": "WARN",
        "detail": f"{len(explore_names)} distinct: {', '.join(sorted(explore_names)) if explore_names else 'none'}",
    }


def check_phase_ordering(strategies):
    """Check 2: All explore_* iterations come before exploit_* iterations."""
    last_explore = -1
    first_exploit = len(strategies)
    for i, s in enumerate(strategies):
        if s.startswith("explore_"):
            last_explore = i
        if s.startswith("exploit_") and i < first_exploit:
            first_exploit = i
    # Ordering is correct if there are no explores, no exploits, or last explore < first exploit
    passed = last_explore < first_exploit or last_explore == -1 or first_exploit == len(strategies)
    return {
        "id": "phase_ordering",
        "name": "explore before exploit ordering",
        "passed": passed,
        "severity": "WARN",
        "detail": f"last_explore={last_explore}, first_exploit={first_exploit if first_exploit < len(strategies) else 'none'}",
    }


def check_no_generic_names(strategies):
    """Check 3: No banned generic strategy names."""
    bad = []
    for s in strategies:
        # Check if the entire strategy name is generic
        if s.lower() in GENERIC_NAMES:
            bad.append(s)
            continue
        # Check if the name part (after phase prefix) is generic
        m = re.match(r'(?:explore|exploit|revert|switch|algebraic)_\d*_?(.+)', s)
        if m and m.group(1).lower() in GENERIC_NAMES:
            bad.append(s)
            continue
        # Check if unprefixed name is generic
        if not PHASE_PATTERN.match(s) and s.lower() in GENERIC_NAMES:
            bad.append(s)
    return {
        "id": "no_generic_names",
        "name": "no banned generic strategy names",
        "passed": len(bad) == 0,
        "severity": "WARN",
        "detail": f"banned: {', '.join(bad)}" if bad else "all descriptive",
    }


def check_min_iterations(strategies):
    """Check 4: At least 2 iterations (explored at least a bit)."""
    return {
        "id": "min_iterations",
        "name": ">=2 iterations",
        "passed": len(strategies) >= 2,
        "severity": "WARN",
        "detail": f"{len(strategies)} iterations",
    }


def check_speedup_consistency(task_dir):
    """Check 5: best_result.json speedup matches progress.json best."""
    best_file = task_dir / "best_result.json"
    progress_file = task_dir / "progress.json"

    if not best_file.exists() or not progress_file.exists():
        return {
            "id": "speedup_consistency",
            "name": "speedup cross-check",
            "passed": True,  # Can't check, skip
            "severity": "WARN",
            "detail": "skipped (missing files)",
        }

    try:
        best = json.loads(best_file.read_text())
        progress = json.loads(progress_file.read_text())
    except json.JSONDecodeError:
        return {
            "id": "speedup_consistency",
            "name": "speedup cross-check",
            "passed": False,
            "severity": "WARN",
            "detail": "JSON parse error",
        }

    best_speedup = best.get("speedup", 0)
    progress_best = progress.get("best", {}).get("speedup", 0)

    # Allow small floating point differences
    passed = abs(best_speedup - progress_best) < 0.01
    return {
        "id": "speedup_consistency",
        "name": "speedup cross-check",
        "passed": passed,
        "severity": "WARN",
        "detail": f"best_result={best_speedup:.3f}x, progress={progress_best:.3f}x" if not passed else f"consistent ({best_speedup:.3f}x)",
    }


def check_section_exists(content, section_text, check_name):
    """Check that a section header/keyword exists in file content."""
    # Case-insensitive search for the section text
    found = section_text.lower() in content.lower()
    return {
        "id": f"section_{check_name.replace(' ', '_')}",
        "name": f"{check_name}",
        "passed": found,
        "severity": "WARN",
        "detail": "found" if found else "missing",
    }


def check_effort_exploration(strategies, best_speedup):
    """Check 6: If task used >=5 iters and best < 1.3x, at least 2 distinct explore strategies."""
    if len(strategies) < 5 or (best_speedup and best_speedup >= 1.3):
        return {
            "id": "effort_exploration",
            "name": "sufficient exploration for hard tasks",
            "passed": True,
            "severity": "WARN",
            "detail": "not applicable (< 5 iters or target met)",
        }

    explore_names = set()
    for s in strategies:
        m = re.match(r'explore_\d+_(.+)', s)
        if m:
            explore_names.add(m.group(1))

    passed = len(explore_names) >= 2
    return {
        "id": "effort_exploration",
        "name": "sufficient exploration for hard tasks",
        "passed": passed,
        "severity": "WARN",
        "detail": f"{len(explore_names)} explore strategies in {len(strategies)} iters (need >=2)" if not passed else f"{len(explore_names)} explore strategies",
    }


# ─── Task Verification ───


def verify_task(task_dir, task_name):
    """Run all checks for one task. Returns (status, checks_detail)."""
    checks = []

    # 1-3: File existence checks
    checks.append(check_file_exists(task_dir / "progress.json", "progress.json", critical=True))
    checks.append(check_file_exists(task_dir / "best_result.json", "best_result.json", critical=True))
    checks.append(check_file_nonempty(task_dir / "reflection.md", "reflection.md", critical=True))

    # 4: algo_trace.md (WARN only)
    checks.append(check_file_nonempty(task_dir / "algo_trace.md", "algo_trace.md", critical=False))

    # Load progress data for phase compliance checks
    strategies = []
    best_speedup = None
    if (task_dir / "progress.json").exists():
        try:
            progress = json.loads((task_dir / "progress.json").read_text())
            strategies = [it.get("strategy", "") for it in progress.get("iterations", [])]
            best = progress.get("best", {})
            best_speedup = best.get("speedup")
        except json.JSONDecodeError:
            pass

    # 5-8: Phase compliance checks
    checks.append(check_explore_diversity(strategies))
    checks.append(check_phase_ordering(strategies))
    checks.append(check_no_generic_names(strategies))
    checks.append(check_min_iterations(strategies))

    # 9: Cross-check
    checks.append(check_speedup_consistency(task_dir))

    # 10-12: reflection.md content checks
    if (task_dir / "reflection.md").exists():
        content = (task_dir / "reflection.md").read_text()
        checks.append(check_section_exists(content, "Op type", "reflection Op type"))
        checks.append(check_section_exists(content, "Bottleneck", "reflection Bottleneck"))
        checks.append(check_section_exists(content, "Exploration summary", "reflection Exploration summary"))
    else:
        # Already flagged as FAIL above, add placeholders
        for name in ["reflection Op type", "reflection Bottleneck", "reflection Exploration summary"]:
            checks.append({"id": f"section_{name.replace(' ', '_')}", "name": name,
                          "passed": False, "severity": "WARN", "detail": "reflection.md missing"})

    # 13-16: algo_trace.md content checks
    if (task_dir / "algo_trace.md").exists():
        content = (task_dir / "algo_trace.md").read_text()
        for section in ["Phase A", "Phase B", "Phase C", "Meta-Observations"]:
            checks.append(check_section_exists(content, section, f"algo_trace {section}"))
    else:
        for section in ["Phase A", "Phase B", "Phase C", "Meta-Observations"]:
            checks.append({"id": f"section_algo_trace_{section.replace(' ', '_')}",
                          "name": f"algo_trace {section}", "passed": False,
                          "severity": "WARN", "detail": "algo_trace.md missing"})

    # 17: Effort check
    checks.append(check_effort_exploration(strategies, best_speedup))

    # Determine overall status
    has_fail = any(c["severity"] == "FAIL" and not c["passed"] for c in checks)
    has_warn = any(c["severity"] == "WARN" and not c["passed"] for c in checks)
    status = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    return status, checks


# ─── Report Generation ───


def generate_report(session_id, session_dir, results):
    """Generate detailed verification markdown report."""
    lines = []
    lines.append("# Kernel Bench Verification Report")
    lines.append("")
    lines.append(f"**Session:** `{session_id}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Summary
    total = len(results)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"**Score:** {pass_count}/{total} PASS, {warn_count} WARN, {fail_count} FAIL")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tasks | {total} |")
    lines.append(f"| PASS | {pass_count} |")
    lines.append(f"| WARN | {warn_count} |")
    lines.append(f"| FAIL | {fail_count} |")
    lines.append("")

    # Top issues (most common failures)
    issue_counts = {}
    for r in results:
        for c in r["checks"]:
            if not c["passed"]:
                key = f"[{c['severity']}] {c['name']}"
                issue_counts[key] = issue_counts.get(key, 0) + 1

    if issue_counts:
        lines.append("## Top Issues")
        lines.append("")
        lines.append("| Issue | Count |")
        lines.append("|-------|-------|")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {issue} | {count} |")
        lines.append("")

    # Per-task results
    lines.append("## Per-Task Results")
    lines.append("")

    # Group by status
    for status_group in ["FAIL", "WARN", "PASS"]:
        group_tasks = [r for r in results if r["status"] == status_group]
        if not group_tasks:
            continue

        lines.append(f"### {status_group} ({len(group_tasks)} tasks)")
        lines.append("")

        for r in sorted(group_tasks, key=lambda x: x["task_name"]):
            lines.append(f"#### {r['task_name']} — {r['status']}")
            lines.append("")
            lines.append("| Check | Status | Detail |")
            lines.append("|-------|--------|--------|")
            for c in r["checks"]:
                mark = "PASS" if c["passed"] else c["severity"]
                lines.append(f"| {c['name']} | {mark} | {c['detail']} |")
            lines.append("")

    return "\n".join(lines)


def print_single_task(task_name, status, checks):
    """Print detailed verification for a single task."""
    print(f"\n=== Verification: {task_name} — {status} ===\n")
    print("| # | Check | Status | Detail |")
    print("|---|-------|--------|--------|")
    for i, c in enumerate(checks, 1):
        mark = "PASS" if c["passed"] else c["severity"]
        print(f"| {i} | {c['name']} | {mark} | {c['detail']} |")
    print()


def main():
    args = sys.argv[1:]

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

    task_filter = args[1] if len(args) > 1 else None

    base = get_output_base()
    session_dir = base / session_id

    if not session_dir.exists():
        print(f"Session not found: {session_id}")
        sys.exit(1)

    # Load manifest to get task list
    manifest_file = session_dir / "session_manifest.json"
    if manifest_file.exists():
        try:
            manifest = json.loads(manifest_file.read_text())
            all_tasks = manifest.get("tasks", [])
        except json.JSONDecodeError:
            print(f"Corrupted manifest for session: {session_id}")
            sys.exit(1)
    else:
        # Fall back to scanning directories
        all_tasks = [d.name for d in session_dir.iterdir()
                     if d.is_dir() and (d / "progress.json").exists()]
        all_tasks.sort()

    if not all_tasks:
        print(f"No tasks found in session: {session_id}")
        sys.exit(1)

    # Filter to completed tasks (have best_result.json) unless single task requested
    if task_filter:
        if task_filter not in all_tasks:
            # Try partial match
            matches = [t for t in all_tasks if task_filter in t]
            if len(matches) == 1:
                task_filter = matches[0]
            elif len(matches) > 1:
                print(f"Ambiguous task name '{task_filter}'. Matches: {', '.join(matches)}")
                sys.exit(1)
            else:
                print(f"Task not found: {task_filter}")
                sys.exit(1)

        task_dir = session_dir / task_filter
        if not task_dir.exists():
            print(f"Task directory not found: {task_filter}")
            sys.exit(1)

        status, checks = verify_task(task_dir, task_filter)
        print_single_task(task_filter, status, checks)
        sys.exit(0)

    # Verify all completed tasks
    results = []
    skipped = 0
    for task_name in all_tasks:
        task_dir = session_dir / task_name

        # Skip tasks without best_result.json (not completed)
        if not task_dir.exists() or not (task_dir / "best_result.json").exists():
            skipped += 1
            continue

        # Skip retryable tasks (server errors, etc.)
        try:
            best = json.loads((task_dir / "best_result.json").read_text())
            if best.get("completion_reason") in SKIP_REASONS:
                skipped += 1
                continue
        except (json.JSONDecodeError, Exception):
            pass

        status, checks = verify_task(task_dir, task_name)
        results.append({
            "task_name": task_name,
            "status": status,
            "checks": checks,
        })

    if not results:
        print(f"No completed tasks to verify in session: {session_id}")
        sys.exit(1)

    # Summary counts
    total = len(results)
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")

    # Console summary
    print(f"\n   ═══ Verification: {session_id} ═══\n")
    print(f"     Verified:    {total} tasks ({skipped} skipped)")
    print(f"     PASS:        {pass_count}")
    print(f"     WARN:        {warn_count}")
    print(f"     FAIL:        {fail_count}")
    print(f"\n     Score: {pass_count}/{total} PASS, {warn_count} WARN, {fail_count} FAIL")
    print()

    # Top issues
    issue_counts = {}
    for r in results:
        for c in r["checks"]:
            if not c["passed"]:
                key = f"[{c['severity']}] {c['name']}"
                issue_counts[key] = issue_counts.get(key, 0) + 1

    if issue_counts:
        print("   ─── Top Issues ───")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"     {count:3d}x  {issue}")
        print()

    # Generate and save report
    report = generate_report(session_id, session_dir, results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = session_dir / f"verification_{timestamp}.md"
    report_file.write_text(report)

    print(f"   Detailed report: {report_file}")
    print()


if __name__ == "__main__":
    main()
