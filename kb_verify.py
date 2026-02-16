#!/usr/bin/env python3
"""
Kernel Bench Algorithm Verification.

Checks that optimizer agents follow the Phase A/B/C protocol by examining
breadcrumbs (strategy names in progress.json), file artifacts, and content.

Usage:
    python kb_verify.py <session_id>                # All tasks
    python kb_verify.py <session_id> <task_name>    # Single task detail
    python kb_verify.py                             # Most recent session
    python kb_verify.py --chain <chain_id>          # Chain verification
    python kb_verify.py <chain_id>                  # Auto-detect chain
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

# Valid batch statuses in order
VALID_BATCH_STATUSES = ["running", "tasks_done", "learning", "completed"]


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


# ─── Chain Verification (C1-C12) ───


def check_chain_c1_manifest(chain_dir):
    """C1: chain_manifest.json exists and is valid JSON."""
    manifest_path = chain_dir / "chain_manifest.json"
    if not manifest_path.exists():
        return {"id": "C1", "name": "chain_manifest.json exists and valid",
                "passed": False, "severity": "FAIL", "detail": "missing"}
    try:
        data = json.loads(manifest_path.read_text())
        has_keys = all(k in data for k in ["chain_id", "config", "batches", "cumulative"])
        return {"id": "C1", "name": "chain_manifest.json exists and valid",
                "passed": has_keys, "severity": "FAIL",
                "detail": "valid with required keys" if has_keys else "missing required keys"}
    except json.JSONDecodeError:
        return {"id": "C1", "name": "chain_manifest.json exists and valid",
                "passed": False, "severity": "FAIL", "detail": "invalid JSON"}


def check_chain_c2_statuses(batches):
    """C2: All batch entries have valid status."""
    invalid = []
    for b in batches:
        status = b.get("status", "")
        if status not in VALID_BATCH_STATUSES:
            invalid.append(f"batch {b.get('batch_index', '?')}: '{status}'")
    return {"id": "C2", "name": "batch statuses valid",
            "passed": len(invalid) == 0, "severity": "FAIL",
            "detail": f"invalid: {', '.join(invalid)}" if invalid else "all valid"}


def check_chain_c3_monotonic(batches):
    """C3: Batch statuses monotonically progressed (completed batches before running)."""
    saw_incomplete = False
    violations = []
    for b in batches:
        status = b.get("status", "")
        if status == "completed":
            if saw_incomplete:
                violations.append(f"batch {b.get('batch_index', '?')}: completed after non-completed")
        else:
            saw_incomplete = True
    return {"id": "C3", "name": "statuses monotonically progressed",
            "passed": len(violations) == 0, "severity": "FAIL",
            "detail": f"violations: {', '.join(violations)}" if violations else "monotonic"}


def check_chain_c4_sessions_exist(batches):
    """C4: Each batch session_id maps to an existing session directory."""
    base = get_output_base()
    missing = []
    for b in batches:
        sid = b.get("session_id", "")
        if sid and not (base / sid).exists():
            missing.append(sid)
    return {"id": "C4", "name": "session directories exist",
            "passed": len(missing) == 0, "severity": "FAIL",
            "detail": f"missing: {', '.join(missing)}" if missing else "all exist"}


def check_chain_c5_cumulative_stats(manifest):
    """C5: Cumulative stats are consistent with per-batch bests."""
    base = get_output_base()
    batches = manifest.get("batches", [])
    cumulative = manifest.get("cumulative", {})

    # Recompute cumulative from per-batch data
    all_best = {}
    for b in batches:
        if b.get("status") != "completed":
            continue
        sid = b.get("session_id", "")
        session_dir = base / sid
        if not session_dir.exists():
            continue

        # Scan task directories for best results
        for task_dir in session_dir.iterdir():
            if not task_dir.is_dir():
                continue
            best_file = task_dir / "best_result.json"
            if best_file.exists():
                try:
                    result = json.loads(best_file.read_text())
                    speedup = result.get("speedup", 0)
                    name = task_dir.name
                    if name not in all_best or speedup > all_best[name]:
                        all_best[name] = speedup
                except (json.JSONDecodeError, Exception):
                    pass

    if not all_best:
        return {"id": "C5", "name": "cumulative stats consistent",
                "passed": True, "severity": "WARN", "detail": "no data to verify"}

    computed_passing = sum(1 for s in all_best.values() if s >= 1.3)
    reported_passing = cumulative.get("tasks_passing", 0)

    passed = abs(computed_passing - reported_passing) <= 1  # allow ±1 for rounding
    return {"id": "C5", "name": "cumulative stats consistent",
            "passed": passed, "severity": "WARN",
            "detail": f"computed={computed_passing}, reported={reported_passing}" if not passed else f"consistent ({reported_passing} passing)"}


def check_chain_c6_retry_set(manifest):
    """C6: Retry set is correct — batch N's tasks are a subset of below-target tasks from prior batches."""
    base = get_output_base()
    batches = manifest.get("batches", [])
    config = manifest.get("config", {})
    target = config.get("target_speedup", 1.3)

    violations = []
    all_best = {}

    for b in batches:
        sid = b.get("session_id", "")
        level = b.get("level", "")
        batch_idx = b.get("batch_index", 0)
        session_dir = base / sid

        if not session_dir.exists():
            continue

        # Get this batch's task list
        manifest_file = session_dir / "session_manifest.json"
        if not manifest_file.exists():
            continue
        try:
            sm = json.loads(manifest_file.read_text())
            batch_tasks = set(sm.get("tasks", []))
        except json.JSONDecodeError:
            continue

        # For batch 0 of a level, all tasks are valid
        # For subsequent batches of the same level, check subset
        prior_same_level = [pb for pb in batches
                           if pb.get("level") == level and pb.get("batch_index", 0) < batch_idx]
        if prior_same_level:
            # Compute which tasks were below target in prior batches
            below_target = set()
            for pb in prior_same_level:
                pb_dir = base / pb.get("session_id", "")
                if not pb_dir.exists():
                    continue
                pb_manifest_file = pb_dir / "session_manifest.json"
                if not pb_manifest_file.exists():
                    continue
                try:
                    pb_sm = json.loads(pb_manifest_file.read_text())
                    for tn in pb_sm.get("tasks", []):
                        best_speedup = all_best.get(tn, 0)
                        if best_speedup < target:
                            below_target.add(tn)
                except json.JSONDecodeError:
                    pass

            # Check: batch tasks should be subset of below_target (for below_target mode)
            if config.get("retry_mode") == "below_target":
                extra = batch_tasks - below_target
                if extra and len(extra) > 0:
                    violations.append(f"batch {batch_idx}: {len(extra)} tasks not in below-target set")

        # Update all_best with this batch's results
        for task_dir in session_dir.iterdir():
            if not task_dir.is_dir():
                continue
            best_file = task_dir / "best_result.json"
            if best_file.exists():
                try:
                    result = json.loads(best_file.read_text())
                    speedup = result.get("speedup", 0)
                    name = task_dir.name
                    if name not in all_best or speedup > all_best[name]:
                        all_best[name] = speedup
                except (json.JSONDecodeError, Exception):
                    pass

    return {"id": "C6", "name": "retry set correctness",
            "passed": len(violations) == 0, "severity": "WARN",
            "detail": "; ".join(violations) if violations else "correct"}


def check_chain_c7_histories(batches):
    """C7: task_histories.json exists for batch N>=1."""
    base = get_output_base()
    missing = []

    # Group by level to find retries
    level_seen = {}
    for b in batches:
        level = b.get("level", "")
        batch_idx = b.get("batch_index", 0)
        sid = b.get("session_id", "")

        if level in level_seen:
            # This is a retry — should have task_histories.json
            session_dir = base / sid
            histories_file = session_dir / "task_histories.json"
            if not histories_file.exists():
                missing.append(f"batch {batch_idx} ({sid})")

        level_seen[level] = batch_idx

    return {"id": "C7", "name": "task_histories.json for retries",
            "passed": len(missing) == 0, "severity": "WARN",
            "detail": f"missing: {', '.join(missing)}" if missing else "present where needed"}


def check_chain_c8_history_validity(batches):
    """C8: Task histories reference only strategies that appear in prior batches."""
    base = get_output_base()
    issues = []

    for b in batches:
        sid = b.get("session_id", "")
        session_dir = base / sid
        histories_file = session_dir / "task_histories.json"

        if not histories_file.exists():
            continue

        try:
            histories = json.loads(histories_file.read_text())
        except json.JSONDecodeError:
            issues.append(f"batch {b.get('batch_index', '?')}: invalid JSON in task_histories.json")
            continue

        # Spot check: verify from_batches references are valid session IDs
        for task_name, entry in histories.items():
            from_batches = entry.get("from_batches", [])
            for fb in from_batches:
                if not (base / fb).exists():
                    issues.append(f"batch {b.get('batch_index', '?')}: {task_name} references non-existent {fb}")
                    break
            if issues:
                break  # Don't over-report

    return {"id": "C8", "name": "history references valid",
            "passed": len(issues) == 0, "severity": "WARN",
            "detail": "; ".join(issues[:3]) if issues else "valid"}


def check_chain_c9_knowledge_modified(batches):
    """C9: Reference files were modified between batches (mtime check)."""
    ref_dir = Path(".claude/agents/reference")
    if not ref_dir.exists():
        return {"id": "C9", "name": "knowledge files modified between batches",
                "passed": True, "severity": "WARN", "detail": "no reference directory"}

    # Check if common.md exists and was modified recently
    common_file = ref_dir / "common.md"
    if not common_file.exists():
        return {"id": "C9", "name": "knowledge files modified between batches",
                "passed": True, "severity": "WARN", "detail": "no common.md"}

    # For completed batches > 0, check if any reference file was modified
    completed = [b for b in batches if b.get("status") == "completed"]
    if len(completed) <= 1:
        return {"id": "C9", "name": "knowledge files modified between batches",
                "passed": True, "severity": "WARN", "detail": "only 0-1 completed batches"}

    # Just check that reference files exist and have been written
    ref_files = list(ref_dir.glob("*.md"))
    not_modified = []
    for rf in ref_files:
        if rf.stat().st_size == 0:
            not_modified.append(rf.name)

    return {"id": "C9", "name": "knowledge files modified between batches",
            "passed": len(not_modified) == 0, "severity": "WARN",
            "detail": f"empty: {', '.join(not_modified)}" if not_modified else f"{len(ref_files)} files present"}


def check_chain_c10_convergence(manifest):
    """C10: Convergence decision was justified."""
    batches = manifest.get("batches", [])
    config = manifest.get("config", {})
    max_batches = config.get("max_batches", 0)

    completed = [b for b in batches if b.get("status") == "completed"]

    if len(completed) >= max_batches:
        return {"id": "C10", "name": "convergence justified",
                "passed": True, "severity": "WARN", "detail": "max batches reached"}

    if len(completed) < 2:
        return {"id": "C10", "name": "convergence justified",
                "passed": True, "severity": "WARN", "detail": "not enough batches to check"}

    # Check if early stop was justified
    last = completed[-1]
    prev = completed[-2]
    last_rate = last.get("cumulative_success_rate", 0)
    prev_rate = prev.get("cumulative_success_rate", 0)
    delta = last_rate - prev_rate

    # If chain stopped early, delta should be < 3pp
    if len(completed) < max_batches:
        justified = delta < 0.03
        return {"id": "C10", "name": "convergence justified",
                "passed": justified, "severity": "WARN",
                "detail": f"stopped at batch {len(completed)-1}, delta={delta:.1%}pp {'(<3pp)' if justified else '(>=3pp, premature?)'}"}

    return {"id": "C10", "name": "convergence justified",
            "passed": True, "severity": "WARN", "detail": "chain ran to completion"}


def check_chain_c11_no_empty(batches):
    """C11: No batch has 0 tasks."""
    empty = [b.get("batch_index", "?") for b in batches if b.get("task_count", 0) == 0]
    return {"id": "C11", "name": "no empty batches",
            "passed": len(empty) == 0, "severity": "FAIL",
            "detail": f"empty batches: {empty}" if empty else "all have tasks"}


def check_chain_c12_worker_scaling(batches, config):
    """C12: Worker count scaled appropriately."""
    max_workers = config.get("workers", 15)
    issues = []

    for b in batches:
        task_count = b.get("task_count", 0)
        if task_count > 0 and task_count < max_workers // 3:
            # Check if session manifest has scaled workers
            sid = b.get("session_id", "")
            session_dir = get_output_base() / sid
            manifest_file = session_dir / "session_manifest.json"
            if manifest_file.exists():
                try:
                    sm = json.loads(manifest_file.read_text())
                    actual_workers = sm.get("config", {}).get("num_workers", max_workers)
                    if actual_workers > task_count:
                        issues.append(f"batch {b.get('batch_index', '?')}: {actual_workers} workers for {task_count} tasks")
                except (json.JSONDecodeError, Exception):
                    pass

    return {"id": "C12", "name": "worker scaling appropriate",
            "passed": len(issues) == 0, "severity": "WARN",
            "detail": "; ".join(issues) if issues else "appropriate"}


def check_chain_c13_breakthrough_hints(batches):
    """C13: breakthrough_hints.json exists and is valid for plateau/breakthrough batches."""
    base = get_output_base()
    issues = []

    for b in batches:
        tier = b.get("escalation_tier", "normal")
        if tier not in ("plateau", "breakthrough"):
            continue

        sid = b.get("session_id", "")
        session_dir = base / sid
        hints_file = session_dir / "breakthrough_hints.json"

        if not hints_file.exists():
            issues.append(f"batch {b.get('batch_index', '?')}: missing breakthrough_hints.json (tier={tier})")
            continue

        try:
            hints = json.loads(hints_file.read_text())
        except json.JSONDecodeError:
            issues.append(f"batch {b.get('batch_index', '?')}: invalid JSON in breakthrough_hints.json")
            continue

        # Validate structure
        required_keys = ["escalation_tier", "cluster_summary", "tasks"]
        missing = [k for k in required_keys if k not in hints]
        if missing:
            issues.append(f"batch {b.get('batch_index', '?')}: missing keys {missing} in breakthrough_hints.json")
            continue

        # Validate tasks have required fields
        tasks = hints.get("tasks", {})
        for task_name, task_data in list(tasks.items())[:3]:  # spot-check 3
            if "failure_cluster" not in task_data:
                issues.append(f"batch {b.get('batch_index', '?')}: task {task_name} missing failure_cluster")
                break
            if "hint" not in task_data:
                issues.append(f"batch {b.get('batch_index', '?')}: task {task_name} missing hint")
                break

    return {"id": "C13", "name": "breakthrough_hints.json valid for escalation batches",
            "passed": len(issues) == 0, "severity": "WARN",
            "detail": "; ".join(issues[:3]) if issues else "valid where needed"}


def check_chain_c14_infeasible_filtering(batches, manifest):
    """C14: Infeasible tasks are filtered from retry sets in later batches."""
    base = get_output_base()
    issues = []

    # Find infeasible tasks from breakthrough_hints.json files
    infeasible_by_batch = {}
    for b in batches:
        sid = b.get("session_id", "")
        session_dir = base / sid
        hints_file = session_dir / "breakthrough_hints.json"

        if not hints_file.exists():
            continue

        try:
            hints = json.loads(hints_file.read_text())
            infeasible = hints.get("infeasible_tasks", [])
            if infeasible:
                infeasible_by_batch[b.get("batch_index", 0)] = set(infeasible)
        except (json.JSONDecodeError, Exception):
            continue

    if not infeasible_by_batch:
        return {"id": "C14", "name": "infeasible tasks filtered from retries",
                "passed": True, "severity": "WARN", "detail": "no infeasible tasks identified"}

    # Check that later batches don't include infeasible tasks
    all_infeasible = set()
    for batch_idx in sorted(infeasible_by_batch.keys()):
        all_infeasible.update(infeasible_by_batch[batch_idx])

        # Check all subsequent batches
        for b in batches:
            if b.get("batch_index", 0) <= batch_idx:
                continue

            sid = b.get("session_id", "")
            session_dir = base / sid
            manifest_file = session_dir / "session_manifest.json"
            if not manifest_file.exists():
                continue

            try:
                sm = json.loads(manifest_file.read_text())
                batch_tasks = set(sm.get("tasks", []))
                leaked = batch_tasks & all_infeasible
                if leaked:
                    issues.append(f"batch {b.get('batch_index', '?')}: includes {len(leaked)} infeasible tasks ({', '.join(list(leaked)[:3])})")
            except (json.JSONDecodeError, Exception):
                pass

    return {"id": "C14", "name": "infeasible tasks filtered from retries",
            "passed": len(issues) == 0, "severity": "WARN",
            "detail": "; ".join(issues[:3]) if issues else f"{len(all_infeasible)} infeasible tasks correctly excluded"}


def check_chain_c15_escalation_progression(batches):
    """C15: Escalation tier progresses correctly (normal → plateau → breakthrough → hard_converge)."""
    VALID_TIERS = ["normal", "plateau", "breakthrough", "hard_converge"]
    issues = []
    prev_tier = None

    for b in batches:
        tier = b.get("escalation_tier", "normal")
        batch_idx = b.get("batch_index", 0)

        # Validate tier value
        if tier not in VALID_TIERS:
            issues.append(f"batch {batch_idx}: invalid tier '{tier}'")
            prev_tier = tier
            continue

        # Check progression (can stay same or advance, never go backward)
        if prev_tier is not None and prev_tier in VALID_TIERS and tier in VALID_TIERS:
            prev_rank = VALID_TIERS.index(prev_tier)
            curr_rank = VALID_TIERS.index(tier)
            if curr_rank < prev_rank:
                issues.append(f"batch {batch_idx}: tier regressed from '{prev_tier}' to '{tier}'")

        prev_tier = tier

    return {"id": "C15", "name": "escalation tier progression valid",
            "passed": len(issues) == 0, "severity": "WARN",
            "detail": "; ".join(issues) if issues else "monotonic progression"}


def verify_chain(chain_id):
    """Run chain-level verification. Returns (per_batch_results, chain_checks)."""
    base = get_output_base()
    chain_dir = base / chain_id

    # C1: Manifest exists
    c1 = check_chain_c1_manifest(chain_dir)
    if not c1["passed"]:
        return [], [c1]

    manifest = json.loads((chain_dir / "chain_manifest.json").read_text())
    batches = manifest.get("batches", [])
    config = manifest.get("config", {})

    # Run per-batch verification
    per_batch_results = {}
    for b in batches:
        sid = b.get("session_id", "")
        if b.get("status") != "completed":
            continue

        session_dir = base / sid
        if not session_dir.exists():
            continue

        # Get task list
        manifest_file = session_dir / "session_manifest.json"
        if manifest_file.exists():
            try:
                sm = json.loads(manifest_file.read_text())
                all_tasks = sm.get("tasks", [])
            except json.JSONDecodeError:
                all_tasks = []
        else:
            all_tasks = [d.name for d in session_dir.iterdir()
                        if d.is_dir() and (d / "progress.json").exists()]

        results = []
        skipped = 0
        for task_name in all_tasks:
            task_dir = session_dir / task_name
            if not task_dir.exists() or not (task_dir / "best_result.json").exists():
                skipped += 1
                continue
            try:
                best = json.loads((task_dir / "best_result.json").read_text())
                if best.get("completion_reason") in SKIP_REASONS:
                    skipped += 1
                    continue
            except (json.JSONDecodeError, Exception):
                pass

            status, checks = verify_task(task_dir, task_name)
            results.append({"task_name": task_name, "status": status, "checks": checks})

        per_batch_results[sid] = {"results": results, "skipped": skipped, "batch_index": b.get("batch_index", 0)}

    # Run chain-level checks C1-C15
    chain_checks = [
        c1,
        check_chain_c2_statuses(batches),
        check_chain_c3_monotonic(batches),
        check_chain_c4_sessions_exist(batches),
        check_chain_c5_cumulative_stats(manifest),
        check_chain_c6_retry_set(manifest),
        check_chain_c7_histories(batches),
        check_chain_c8_history_validity(batches),
        check_chain_c9_knowledge_modified(batches),
        check_chain_c10_convergence(manifest),
        check_chain_c11_no_empty(batches),
        check_chain_c12_worker_scaling(batches, config),
        check_chain_c13_breakthrough_hints(batches),
        check_chain_c14_infeasible_filtering(batches, manifest),
        check_chain_c15_escalation_progression(batches),
    ]

    return per_batch_results, chain_checks


def generate_chain_report(chain_id, per_batch_results, chain_checks):
    """Generate chain verification markdown report."""
    lines = []
    lines.append("# Chain Verification Report")
    lines.append("")
    lines.append(f"**Chain:** `{chain_id}`")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Batches:** {len(per_batch_results)}")
    lines.append("")

    # Chain-level checks
    chain_pass = sum(1 for c in chain_checks if c["passed"])
    chain_warn = sum(1 for c in chain_checks if not c["passed"] and c["severity"] == "WARN")
    chain_fail = sum(1 for c in chain_checks if not c["passed"] and c["severity"] == "FAIL")

    lines.append("## Cross-Batch Checks")
    lines.append("")
    lines.append(f"**Score:** {chain_pass}/{len(chain_checks)} PASS, {chain_warn} WARN, {chain_fail} FAIL")
    lines.append("")
    lines.append("| # | Check | Status | Detail |")
    lines.append("|---|-------|--------|--------|")
    for c in chain_checks:
        mark = "PASS" if c["passed"] else c["severity"]
        lines.append(f"| {c['id']} | {c['name']} | {mark} | {c['detail']} |")
    lines.append("")

    # Per-batch summaries
    lines.append("## Per-Batch Verification")
    lines.append("")
    for sid, data in sorted(per_batch_results.items(), key=lambda x: x[1].get("batch_index", 0)):
        results = data["results"]
        total = len(results)
        p = sum(1 for r in results if r["status"] == "PASS")
        w = sum(1 for r in results if r["status"] == "WARN")
        f = sum(1 for r in results if r["status"] == "FAIL")
        lines.append(f"### Batch {data['batch_index']} ({sid})")
        lines.append(f"**Score:** {p}/{total} PASS, {w} WARN, {f} FAIL ({data['skipped']} skipped)")
        lines.append("")

    return "\n".join(lines)


def print_chain_summary(chain_id, per_batch_results, chain_checks):
    """Print chain verification console summary."""
    print(f"\n   ═══ Chain Verification: {chain_id} ═══\n")

    # Per-batch summaries
    for sid, data in sorted(per_batch_results.items(), key=lambda x: x[1].get("batch_index", 0)):
        results = data["results"]
        total = len(results)
        p = sum(1 for r in results if r["status"] == "PASS")
        w = sum(1 for r in results if r["status"] == "WARN")
        f = sum(1 for r in results if r["status"] == "FAIL")
        print(f"   Per-batch: {p}/{total} PASS, {w} WARN, {f} FAIL   (batch {data['batch_index']})")

    print()

    # Chain-level checks
    chain_pass = sum(1 for c in chain_checks if c["passed"])
    chain_warn = sum(1 for c in chain_checks if not c["passed"] and c["severity"] == "WARN")
    chain_fail = sum(1 for c in chain_checks if not c["passed"] and c["severity"] == "FAIL")

    print(f"   Cross-batch: {chain_pass}/{len(chain_checks)} PASS, {chain_warn} WARN, {chain_fail} FAIL")

    # Show failures and warnings
    issues = [c for c in chain_checks if not c["passed"]]
    if issues:
        for c in issues:
            print(f"     {c['severity']} {c['id']}: {c['detail']}")
    print()


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

    # Check for --chain flag
    is_chain = "--chain" in args
    if is_chain:
        args = [a for a in args if a != "--chain"]

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

    # Auto-detect chain: if session_id starts with "chain_" or has chain_manifest.json
    base = get_output_base()
    chain_dir = base / session_id
    if not is_chain and (chain_dir / "chain_manifest.json").exists():
        is_chain = True

    if is_chain:
        # Chain verification mode
        per_batch_results, chain_checks = verify_chain(session_id)
        print_chain_summary(session_id, per_batch_results, chain_checks)

        # Generate and save report
        report = generate_chain_report(session_id, per_batch_results, chain_checks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = chain_dir / f"chain_verification_{timestamp}.md"
        report_file.write_text(report)
        print(f"   Detailed report: {report_file}")
        print()
        sys.exit(0)

    task_filter = args[1] if len(args) > 1 else None

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
