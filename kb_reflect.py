#!/usr/bin/env python3
"""
Aggregate reflection.md files from kernel-bench sessions into per-op-type learned files.

Usage:
    python kb_reflect.py <session_id>       # Aggregate one session
    python kb_reflect.py                    # Uses most recent session

Output:
    .claude/agents/learned/{op_type}.md     # One file per op type
    Each file capped at MAX_PER_OP reflections, keeping highest speedup.

Merges with existing files — safe to run multiple times across sessions.
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime

MAX_PER_OP = 5  # Max reflections kept per op type


def get_output_base():
    return Path(os.path.expanduser("~/.inference/claude_code_output"))


def get_most_recent_session():
    base = get_output_base()
    if not base.exists():
        return None
    sessions = [d for d in base.iterdir() if d.is_dir() and (d / "session_manifest.json").exists()]
    if not sessions:
        return None
    return max(sessions, key=lambda d: d.stat().st_mtime).name


def parse_reflection(text):
    """Parse a single reflection entry. Returns (task_name, op_type, speedup, text)."""
    op_match = re.search(r'\*\*Op type\*\*:\s*(.+?)(?:\n|$)', text)
    op_type = op_match.group(1).strip().rstrip(',').split()[0].lower() if op_match else "other"
    # Normalize compound types like "conv + normalization" to just "conv"
    op_type = op_type.split('+')[0].strip()

    speed_match = re.search(r'###\s*(.+?)\s*\((\d+\.?\d*)x', text)
    if speed_match:
        task_name = speed_match.group(1).strip()
        speedup = float(speed_match.group(2))
    else:
        task_name = "unknown"
        speedup = 0.0

    return task_name, op_type, speedup, text.strip()


def split_into_entries(text):
    """Split a file into individual reflection entries (each starts with ###)."""
    entries = []
    current = []
    for line in text.split('\n'):
        if line.startswith('### ') and current:
            entries.append('\n'.join(current))
            current = [line]
        elif line.startswith('# Reflection:') and current:
            # Handle non-standard format (e.g., "# Reflection: task_name")
            entries.append('\n'.join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append('\n'.join(current))
    return [e.strip() for e in entries if e.strip()]


def aggregate(session_id):
    session_dir = get_output_base() / session_id
    if not session_dir.exists():
        print(f"Session not found: {session_id}")
        sys.exit(1)

    # Find all reflection.md files
    reflection_files = sorted(session_dir.glob("*/reflection.md"))
    if not reflection_files:
        print(f"No reflection.md files found in {session_id}")
        sys.exit(1)

    # Parse new reflections from this session
    # Key: (op_type) -> list of (task_name, speedup, text)
    new_entries = {}
    for path in reflection_files:
        raw = path.read_text().strip()
        if not raw:
            continue
        for entry_text in split_into_entries(raw):
            task_name, op_type, speedup, text = parse_reflection(entry_text)
            new_entries.setdefault(op_type, []).append((task_name, speedup, text))

    # Determine target directory
    script_dir = Path(__file__).resolve().parent
    learned_dir = script_dir / ".claude" / "agents" / "learned"
    learned_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for op_type, entries in new_entries.items():
        op_file = learned_dir / f"{op_type}.md"

        # Load existing entries from this op file
        existing = {}  # task_name -> (speedup, text)
        if op_file.exists():
            for entry_text in split_into_entries(op_file.read_text()):
                t_name, _, t_speed, t_text = parse_reflection(entry_text)
                # Keep the better speedup if duplicate task
                if t_name not in existing or t_speed > existing[t_name][0]:
                    existing[t_name] = (t_speed, t_text)

        # Merge new entries (keep better speedup per task)
        for task_name, speedup, text in entries:
            if task_name not in existing or speedup > existing[task_name][0]:
                existing[task_name] = (speedup, text)

        # Sort by speedup descending, cap at MAX_PER_OP
        sorted_entries = sorted(existing.values(), key=lambda x: x[0], reverse=True)
        kept = sorted_entries[:MAX_PER_OP]
        dropped = len(sorted_entries) - len(kept)

        # Write the file
        header = f"# {op_type} patterns\n<!-- Updated: {datetime.now().strftime('%Y-%m-%d')} | Top {len(kept)} by speedup -->\n"
        body = "\n\n".join(text for _, text in kept)
        op_file.write_text(f"{header}\n{body}\n")

        stats[op_type] = (len(kept), dropped)

    # Summary
    total_kept = sum(k for k, _ in stats.values())
    total_dropped = sum(d for _, d in stats.values())
    print(f"Aggregated reflections from {session_id} → .claude/agents/learned/")
    for op_type, (kept, dropped) in sorted(stats.items()):
        suffix = f" (dropped {dropped} lowest)" if dropped else ""
        print(f"  {op_type}.md: {kept} reflections{suffix}")
    print(f"  Total: {total_kept} kept, {total_dropped} dropped")


def main():
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        session_id = get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
        print(f"Using most recent session: {session_id}")

    aggregate(session_id)


if __name__ == "__main__":
    main()
