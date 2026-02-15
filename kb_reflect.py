#!/usr/bin/env python3
"""
Aggregate and collect reflection.md and algo_trace.md files from kernel-bench sessions.

Usage:
    python kb_reflect.py <session_id>                    # Collect reflections for learning agent
    python kb_reflect.py collect <session_id>             # Same as above (explicit)
    python kb_reflect.py collect_traces <session_id>      # Collect algo traces into all_algo_traces.md
    python kb_reflect.py classify <session_id>            # Split reflections into per-op-type files
    python kb_reflect.py aggregate <session_id>           # Legacy: mechanical top-5-by-speedup aggregation

The default mode ('collect') concatenates all reflection.md files into
all_reflections.md for the learning agent to process. The 'aggregate' mode
is the legacy mechanical aggregation (top-5-by-speedup per op type).

Output (collect mode):
    ~/.inference/claude_code_output/{session_id}/all_reflections.md

Output (collect_traces mode):
    ~/.inference/claude_code_output/{session_id}/all_algo_traces.md

Output (classify mode):
    ~/.inference/claude_code_output/{session_id}/reflections_by_op/{op_type}.md

Output (aggregate mode):
    .claude/agents/learned/{op_type}.md
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime

MAX_PER_OP = 5  # Max reflections kept per op type (aggregate mode only)


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


def collect(session_id):
    """Concatenate all reflection.md files into all_reflections.md for learning agent."""
    session_dir = get_output_base() / session_id
    if not session_dir.exists():
        print(f"Session not found: {session_id}")
        sys.exit(1)

    # Find all reflection.md files
    reflection_files = sorted(session_dir.glob("*/reflection.md"))
    if not reflection_files:
        print(f"No reflection.md files found in {session_id}")
        sys.exit(1)

    # Concatenate all reflections
    all_reflections = []
    for path in reflection_files:
        raw = path.read_text().strip()
        if raw:
            all_reflections.append(raw)

    output_path = session_dir / "all_reflections.md"
    header = f"# All Reflections — {session_id}\n<!-- {len(reflection_files)} reflection files, collected {datetime.now().strftime('%Y-%m-%d %H:%M')} -->\n"
    output_path.write_text(header + "\n" + "\n\n---\n\n".join(all_reflections) + "\n")

    print(f"Collected {len(reflection_files)} reflections → {output_path}")
    return str(output_path)


def collect_traces(session_id):
    """Concatenate all algo_trace.md files into all_algo_traces.md for algorithm learner."""
    session_dir = get_output_base() / session_id
    if not session_dir.exists():
        print(f"Session not found: {session_id}")
        sys.exit(1)

    # Find all algo_trace.md files
    trace_files = sorted(session_dir.glob("*/algo_trace.md"))
    if not trace_files:
        print(f"No algo_trace.md files found in {session_id}")
        sys.exit(1)

    # Concatenate all traces
    all_traces = []
    for path in trace_files:
        raw = path.read_text().strip()
        if raw:
            all_traces.append(raw)

    output_path = session_dir / "all_algo_traces.md"
    header = f"# All Algorithm Traces — {session_id}\n<!-- {len(trace_files)} trace files, collected {datetime.now().strftime('%Y-%m-%d %H:%M')} -->\n"
    output_path.write_text(header + "\n" + "\n\n---\n\n".join(all_traces) + "\n")

    print(f"Collected {len(trace_files)} algo traces → {output_path}")
    return str(output_path)


def classify(session_id):
    """Split all_reflections.md into per-op-type files for parallel kernel learning."""
    session_dir = get_output_base() / session_id
    if not session_dir.exists():
        print(f"Session not found: {session_id}")
        sys.exit(1)

    # Ensure all_reflections.md exists
    all_ref_path = session_dir / "all_reflections.md"
    if not all_ref_path.exists():
        print(f"all_reflections.md not found. Run 'collect' first.")
        sys.exit(1)

    raw = all_ref_path.read_text()

    # Split into individual reflection entries
    entries = split_into_entries(raw)

    # Classify by op type
    by_op = {}  # op_type -> list of entry texts
    for entry_text in entries:
        _, op_type, _, text = parse_reflection(entry_text)
        # Skip header-only entries (from the all_reflections.md header line)
        if op_type == "other" and text.startswith("# All Reflections"):
            continue
        by_op.setdefault(op_type, []).append(text)

    # Write per-op-type files
    output_dir = session_dir / "reflections_by_op"
    output_dir.mkdir(exist_ok=True)

    populated = []
    for op_type, op_entries in sorted(by_op.items()):
        if not op_entries:
            continue
        # Sanitize op_type for filename (replace / with _)
        safe_op_type = op_type.replace("/", "_")
        op_file = output_dir / f"{safe_op_type}.md"
        header = f"# {op_type} Reflections — {session_id}\n<!-- {len(op_entries)} entries, classified {datetime.now().strftime('%Y-%m-%d %H:%M')} -->\n"
        op_file.write_text(header + "\n" + "\n\n---\n\n".join(op_entries) + "\n")
        populated.append(safe_op_type)

    # Print summary (used by skill controller to know which agents to spawn)
    print(f"Classified {sum(len(v) for v in by_op.values())} reflections into {len(populated)} op types")
    for op_type, op_entries in sorted(by_op.items()):
        if op_entries:
            safe = op_type.replace("/", "_")
            print(f"  {safe}: {len(op_entries)} reflections")
    print(f"POPULATED_OPS: {','.join(populated)}")
    return populated


def aggregate(session_id):
    """Legacy: mechanical top-5-by-speedup aggregation into per-op-type files."""
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
    args = sys.argv[1:]

    if not args:
        session_id = get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
        print(f"Using most recent session: {session_id}")
        collect(session_id)
        return

    # Check for subcommand
    if args[0] == "collect":
        session_id = args[1] if len(args) > 1 else get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
        collect(session_id)
    elif args[0] == "collect_traces":
        session_id = args[1] if len(args) > 1 else get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
        collect_traces(session_id)
    elif args[0] == "classify":
        session_id = args[1] if len(args) > 1 else get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
        collect(session_id)  # Ensure all_reflections.md exists first
        classify(session_id)
    elif args[0] == "aggregate":
        session_id = args[1] if len(args) > 1 else get_most_recent_session()
        if not session_id:
            print("No sessions found.")
            sys.exit(1)
        aggregate(session_id)
    else:
        # Default: treat first arg as session_id, default to collect mode
        collect(args[0])


if __name__ == "__main__":
    main()
