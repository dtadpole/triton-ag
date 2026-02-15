#!/usr/bin/env python3
"""
Kernel Bench Task History Generator.

Generates task_histories.json for a chain batch from prior batch data.
This file is consumed by optimizer agents to avoid repeating strategies
across batches in a chain.

Usage:
    python3 kb_history.py <chain_dir> <batch_index> <session_id>

Arguments:
    chain_dir     Path to the chain directory (contains chain_manifest.json)
    batch_index   Index of the current batch (histories are built from batches 0..N-1)
    session_id    Session ID for the current batch (output is written to its session dir)

Output:
    Writes task_histories.json to ~/.inference/claude_code_output/{session_id}/
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime


def get_output_base():
    return Path(os.path.expanduser("~/.inference/claude_code_output"))


def load_chain_manifest(chain_dir):
    """Load and validate chain_manifest.json."""
    manifest_path = Path(chain_dir) / "chain_manifest.json"
    if not manifest_path.exists():
        print(f"Error: chain_manifest.json not found in {chain_dir}")
        sys.exit(1)
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        print(f"Error: corrupted chain_manifest.json in {chain_dir}")
        sys.exit(1)


def load_task_progress(session_dir, task_name):
    """Load progress.json for a specific task in a session."""
    progress_file = session_dir / task_name / "progress.json"
    if not progress_file.exists():
        return None
    try:
        return json.loads(progress_file.read_text())
    except json.JSONDecodeError:
        return None


def load_reflection_summary(session_dir, task_name):
    """Extract a one-line summary from reflection.md."""
    reflection_file = session_dir / task_name / "reflection.md"
    if not reflection_file.exists():
        return None
    try:
        content = reflection_file.read_text()
    except Exception:
        return None

    # Extract "Key insight" line
    for line in content.split("\n"):
        if "key insight" in line.lower():
            # Remove markdown bold markers and the label
            summary = re.sub(r'\*\*Key [Ii]nsight\*\*:\s*', '', line).strip()
            if summary:
                return summary

    # Fallback: first non-header, non-empty line
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("**"):
            return stripped[:200]

    return None


def format_strategy_entry(iteration):
    """Format a single iteration into a strategy_tried entry string."""
    strategy = iteration.get("strategy", "unknown")
    speedup = iteration.get("speedup", 0)

    if not iteration.get("compiled", False):
        status = "compile_error"
    elif not iteration.get("correct", False):
        status = "incorrect"
    else:
        status = "correct"

    return f"{strategy} ({speedup:.2f}x, {status})"


def build_task_histories(manifest, batch_index):
    """Build task histories dict from all prior batches."""
    base = get_output_base()
    histories = {}

    # Gather data from all prior batches (index < batch_index)
    prior_batches = [b for b in manifest.get("batches", [])
                     if b.get("batch_index", 0) < batch_index]

    if not prior_batches:
        return histories

    for batch in prior_batches:
        batch_session_id = batch.get("session_id")
        if not batch_session_id:
            continue

        session_dir = base / batch_session_id
        if not session_dir.exists():
            continue

        # Get task list from session manifest
        session_manifest_file = session_dir / "session_manifest.json"
        if session_manifest_file.exists():
            try:
                session_manifest = json.loads(session_manifest_file.read_text())
                task_names = session_manifest.get("tasks", [])
            except json.JSONDecodeError:
                task_names = []
        else:
            # Fall back to scanning directories
            task_names = [d.name for d in session_dir.iterdir()
                         if d.is_dir() and (d / "progress.json").exists()]

        for task_name in task_names:
            progress = load_task_progress(session_dir, task_name)
            if not progress:
                continue

            iterations = progress.get("iterations", [])
            best = progress.get("best", {})
            best_speedup = best.get("speedup", 0)
            best_strategy = best.get("strategy", "")

            # Build strategy entries from iterations
            strategy_entries = [format_strategy_entry(it) for it in iterations]

            # Load reflection summary
            reflection_summary = load_reflection_summary(session_dir, task_name)

            if task_name not in histories:
                histories[task_name] = {
                    "previous_best_speedup": best_speedup,
                    "previous_best_strategy": best_strategy,
                    "strategies_tried": strategy_entries,
                    "reflection_summary": reflection_summary,
                    "from_batches": [batch_session_id],
                }
            else:
                # Merge with existing entry from earlier batch
                entry = histories[task_name]

                # Keep highest speedup
                if best_speedup > entry["previous_best_speedup"]:
                    entry["previous_best_speedup"] = best_speedup
                    entry["previous_best_strategy"] = best_strategy

                # Append strategies (dedup not needed — different batches = different attempts)
                entry["strategies_tried"].extend(strategy_entries)

                # Use most recent reflection summary
                if reflection_summary:
                    entry["reflection_summary"] = reflection_summary

                # Track which batches contributed
                entry["from_batches"].append(batch_session_id)

    return histories


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)

    chain_dir = sys.argv[1]
    try:
        batch_index = int(sys.argv[2])
    except ValueError:
        print(f"Error: batch_index must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    session_id = sys.argv[3]

    if batch_index < 1:
        print(f"Error: batch_index must be >= 1 (batch 0 has no prior history)")
        sys.exit(1)

    # Load chain manifest
    manifest = load_chain_manifest(chain_dir)

    # Build histories from prior batches
    histories = build_task_histories(manifest, batch_index)

    if not histories:
        print(f"No task histories found from prior batches (0..{batch_index-1})")
        sys.exit(0)

    # Write to session directory
    base = get_output_base()
    session_dir = base / session_id
    if not session_dir.exists():
        print(f"Error: session directory not found: {session_dir}")
        sys.exit(1)

    output_file = session_dir / "task_histories.json"
    output_file.write_text(json.dumps(histories, indent=2))

    print(f"Wrote {len(histories)} task histories to {output_file}")


if __name__ == "__main__":
    main()
