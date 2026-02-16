#!/usr/bin/env python3
"""
Kernel Bench Breakthrough Analysis.

Analyzes stuck tasks from prior chain batches, clusters them by failure mode,
finds cross-task transfer opportunities, and generates breakthrough_hints.json
for the next batch's optimizers.

This script is invoked by the chain orchestrator when the escalation tier
reaches PLATEAU or BREAKTHROUGH. It replaces the naive "don't repeat strategies"
approach with targeted, per-task guidance.

Usage:
    python3 kb_breakthrough.py <chain_dir> <batch_index> <session_id>

Arguments:
    chain_dir     Path to the chain directory (contains chain_manifest.json)
    batch_index   Index of the upcoming batch (analysis covers batches 0..N-1)
    session_id    Session ID for the upcoming batch (output written to its dir)

Output:
    Writes breakthrough_hints.json to ~/.inference/claude_code_output/{session_id}/
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime


# ─── Constants ───

# Failure cluster thresholds
CLOSE_TO_TARGET_MIN = 1.1
TARGET_SPEEDUP = 1.3
INFEASIBLE_MAX = 0.2
INFEASIBLE_MIN_BATCHES = 2
INFEASIBLE_MIN_ITERS = 10

# Known environment ceilings (patterns in task names that are likely infeasible)
KNOWN_CEILINGS = {
    "LSTM": "RNN/LSTM requires cuDNN fused ops, Triton cannot match",
    "GRU": "GRU requires cuDNN fused ops at training=True",
}


def get_output_base():
    return Path(os.path.expanduser("~/.inference/claude_code_output"))


def load_chain_manifest(chain_dir):
    manifest_path = Path(chain_dir) / "chain_manifest.json"
    if not manifest_path.exists():
        print(f"Error: chain_manifest.json not found in {chain_dir}")
        sys.exit(1)
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        print(f"Error: corrupted chain_manifest.json in {chain_dir}")
        sys.exit(1)


# ─── Data Collection ───


def collect_task_data(manifest, batch_index):
    """Collect per-task data from all prior batches.

    Returns:
        all_tasks: dict of task_name -> {
            best_speedup, best_strategy, best_session, best_iteration,
            total_iterations, iterations_by_batch, reflections, op_type
        }
        passing_tasks: dict of task_name -> same structure (for tasks >= 1.3x)
    """
    base = get_output_base()
    all_tasks = {}

    prior_batches = [b for b in manifest.get("batches", [])
                     if b.get("batch_index", 0) < batch_index
                     and b.get("status") == "completed"]

    for batch in prior_batches:
        sid = batch.get("session_id", "")
        batch_idx = batch.get("batch_index", 0)
        session_dir = base / sid

        if not session_dir.exists():
            continue

        # Get task list
        manifest_file = session_dir / "session_manifest.json"
        if manifest_file.exists():
            try:
                sm = json.loads(manifest_file.read_text())
                task_names = sm.get("tasks", [])
            except json.JSONDecodeError:
                continue
        else:
            task_names = [d.name for d in session_dir.iterdir()
                         if d.is_dir() and (d / "progress.json").exists()]

        for task_name in task_names:
            task_dir = session_dir / task_name
            if not task_dir.exists():
                continue

            # Load progress
            progress_file = task_dir / "progress.json"
            if not progress_file.exists():
                continue
            try:
                progress = json.loads(progress_file.read_text())
            except json.JSONDecodeError:
                continue

            iterations = progress.get("iterations", [])
            best = progress.get("best", {})
            best_speedup = best.get("speedup", 0)
            best_strategy = best.get("strategy", "")
            best_iter = best.get("iteration", -1)

            # Extract op_type from reflection if available
            op_type = None
            reflection_file = task_dir / "reflection.md"
            if reflection_file.exists():
                try:
                    content = reflection_file.read_text()
                    # Look for "Op type: xxx" or "**Op type**: xxx"
                    m = re.search(r'\*?\*?Op type\*?\*?:?\s*(\S+)', content, re.IGNORECASE)
                    if m:
                        op_type = m.group(1).strip().lower().rstrip(",.")
                except Exception:
                    pass

            # Initialize or update task entry
            if task_name not in all_tasks:
                all_tasks[task_name] = {
                    "best_speedup": 0,
                    "best_strategy": "",
                    "best_session": "",
                    "best_iteration": -1,
                    "total_iterations": 0,
                    "iterations_by_batch": {},
                    "strategies_tried": [],
                    "compile_failures": 0,
                    "correctness_failures": 0,
                    "correct_count": 0,
                    "op_type": None,
                    "batches_seen": 0,
                    "reflection_summary": None,
                    "best_kernel_path": None,
                }

            entry = all_tasks[task_name]
            entry["batches_seen"] += 1

            # Update best
            if best_speedup > entry["best_speedup"]:
                entry["best_speedup"] = best_speedup
                entry["best_strategy"] = best_strategy
                entry["best_session"] = sid
                entry["best_iteration"] = best_iter
                # Record path to best kernel
                kernel_file = task_dir / f"iteration_{best_iter:02d}_cuda_kernel.py"
                if kernel_file.exists():
                    entry["best_kernel_path"] = str(kernel_file)

            # Accumulate iteration stats
            entry["total_iterations"] += len(iterations)
            entry["iterations_by_batch"][batch_idx] = len(iterations)

            for it in iterations:
                strategy = it.get("strategy", "unknown")
                entry["strategies_tried"].append(strategy)
                if not it.get("compiled", False):
                    entry["compile_failures"] += 1
                elif not it.get("correct", False):
                    entry["correctness_failures"] += 1
                else:
                    entry["correct_count"] += 1

            if op_type:
                entry["op_type"] = op_type

            # Grab most recent reflection summary
            if reflection_file.exists():
                try:
                    content = reflection_file.read_text()
                    for line in content.split("\n"):
                        if "key insight" in line.lower():
                            summary = re.sub(r'\*\*Key [Ii]nsight\*\*:\s*', '', line).strip()
                            if summary:
                                entry["reflection_summary"] = summary
                                break
                except Exception:
                    pass

    # Split into passing and stuck
    passing_tasks = {name: data for name, data in all_tasks.items()
                     if data["best_speedup"] >= TARGET_SPEEDUP}

    return all_tasks, passing_tasks


# ─── Failure Clustering ───


def classify_failure(task_name, data):
    """Classify a stuck task into a failure cluster."""
    best = data["best_speedup"]
    total = data["total_iterations"]
    compile_rate = data["compile_failures"] / total if total > 0 else 0
    correct_rate = data["correctness_failures"] / total if total > 0 else 0
    batches = data["batches_seen"]

    # Infeasible: consistently very low across multiple batches
    if best < INFEASIBLE_MAX and batches >= INFEASIBLE_MIN_BATCHES and total >= INFEASIBLE_MIN_ITERS:
        return "infeasible"

    # Check known ceilings
    for pattern, reason in KNOWN_CEILINGS.items():
        if pattern.lower() in task_name.lower():
            if best < 1.0 and batches >= 2:
                return "infeasible"

    # Close to target: real progress, just needs tuning
    if best >= CLOSE_TO_TARGET_MIN:
        return "close_to_target"

    # Correctness stuck: most iterations compile but fail correctness
    if compile_rate < 0.3 and correct_rate >= 0.5:
        return "correctness_stuck"

    # Compile stuck: most iterations fail to compile
    if compile_rate >= 0.5:
        return "compile_stuck"

    # Performance ceiling: compiles and runs correctly but too slow
    return "perf_ceiling"


# ─── Cross-Task Transfer ───


def find_similar_passing_task(task_name, task_data, passing_tasks):
    """Find the most structurally similar passing task.

    Matching priority:
    1. Same op_type
    2. Similar task name pattern (shared numeric/keyword prefixes)
    3. Highest speedup among matches
    """
    task_op = task_data.get("op_type")
    candidates = []

    for pass_name, pass_data in passing_tasks.items():
        score = 0

        # Op type match (strongest signal)
        if task_op and pass_data.get("op_type") == task_op:
            score += 10

        # Name similarity (shared words after removing numbers)
        task_words = set(re.findall(r'[A-Za-z]+', task_name.lower()))
        pass_words = set(re.findall(r'[A-Za-z]+', pass_name.lower()))
        shared = task_words & pass_words - {"relu", "sigmoid", "tanh", "softmax"}  # skip trivial common words
        score += len(shared) * 2

        if score > 0:
            candidates.append((score, pass_data["best_speedup"], pass_name, pass_data))

    if not candidates:
        return None

    # Sort by score desc, then speedup desc
    candidates.sort(key=lambda x: (-x[0], -x[1]))
    _, _, best_name, best_data = candidates[0]

    return {
        "task_name": best_name,
        "speedup": best_data["best_speedup"],
        "strategy": best_data["best_strategy"],
        "kernel_path": best_data.get("best_kernel_path"),
        "op_type": best_data.get("op_type"),
    }


def find_untried_strategies(task_data, op_type):
    """Identify reference strategies not yet tried.

    Compares strategies_tried against known strategy families from reference files.
    Returns strategy families that were never attempted.
    """
    tried = set()
    for s in task_data.get("strategies_tried", []):
        # Extract the strategy family (strip explore_N_, exploit_N_ prefix)
        m = re.match(r'(?:explore|exploit|revert|switch|algebraic)_\d+_(.+)', s)
        if m:
            tried.add(m.group(1).lower())
        else:
            tried.add(s.lower())

    # Known strategy families per op type
    strategy_families = {
        "element_wise": ["vectorized_loads", "fused_chain", "block_reduce_fused", "fp16_accumulate"],
        "matmul": ["tiled_matmul", "epilogue_fusion", "split_k", "persistent_kernel", "fp16_tensor_cores"],
        "conv": ["implicit_gemm", "fp16_nobias", "cudnn_passthrough", "fused_postops", "winograd"],
        "reduction": ["tree_reduce", "welford", "online_softmax", "two_pass", "warp_reduce"],
        "normalization": ["fused_norm_act", "parallel_stats", "welford_fused", "group_norm_fused"],
        "loss": ["online_loss", "fused_softmax_ce", "numerically_stable"],
        "pooling": ["fused_pool_act", "block_pool", "adaptive_grid"],
        "other": ["fused_pipeline", "fp16_pipeline", "cudnn_hybrid", "aten_delegation"],
    }

    available = strategy_families.get(op_type, strategy_families["other"])
    untried = [s for s in available if not any(s in t for t in tried)]

    return untried


# ─── Hint Generation ───


def generate_hint(task_name, task_data, cluster, similar, untried):
    """Generate a natural language hint for the optimizer."""
    best = task_data["best_speedup"]

    if cluster == "close_to_target":
        if similar:
            return (f"{similar['task_name']} achieved {similar['speedup']:.2f}x with "
                    f"{similar['strategy']}. Your task has similar structure (both {task_data.get('op_type', 'unknown')} ops). "
                    f"Read its kernel code and adapt — focus on block size tuning and memory layout, "
                    f"not new algorithms. Current best is {best:.2f}x, need {TARGET_SPEEDUP}x.")
        return (f"Current best {best:.2f}x is close to {TARGET_SPEEDUP}x target. "
                f"Focus on micro-tuning: block sizes, num_warps, memory coalescing, "
                f"vectorized loads. Do NOT try new algorithms — tune the best existing approach.")

    if cluster == "perf_ceiling":
        if untried:
            return (f"All prior approaches plateaued at {best:.2f}x. "
                    f"Untried strategy families: {', '.join(untried[:3])}. "
                    f"Try a fundamentally different kernel architecture, not parameter tuning.")
        if similar:
            return (f"All standard approaches exhausted ({best:.2f}x best). "
                    f"{similar['task_name']} (similar ops) passed with {similar['strategy']}. "
                    f"Read its kernel and try adapting that structure. "
                    f"Consider combining elements from your two best-performing approaches.")
        return (f"All standard approaches exhausted ({best:.2f}x best). "
                f"Try: different parallelism axis, different fusion grouping, "
                f"or hybrid approach (aten for bottleneck op + Triton for the rest).")

    if cluster == "correctness_stuck":
        return (f"Correctness failures dominate ({task_data['correctness_failures']}/{task_data['total_iterations']} iterations). "
                f"Write the simplest correct kernel first — minimal optimizations, straightforward "
                f"indexing, small block sizes. Once correct, THEN tune for speed.")

    if cluster == "compile_stuck":
        return (f"Compilation failures dominate ({task_data['compile_failures']}/{task_data['total_iterations']} iterations). "
                f"Likely Triton API issue. Check: tl.arange must be power-of-2, "
                f"tl.dot needs M,N,K >= 16, avoid tl.static_range > 50 iterations. "
                f"Start with the simplest possible kernel that compiles.")

    if cluster == "infeasible":
        return (f"Likely infeasible: {best:.2f}x across {task_data['batches_seen']} batches, "
                f"{task_data['total_iterations']} total iterations. Environment ceiling.")

    return ""


# ─── Main Logic ───


def build_breakthrough_hints(manifest, batch_index):
    """Build breakthrough hints from all prior batch data."""
    all_tasks, passing_tasks = collect_task_data(manifest, batch_index)

    if not all_tasks:
        return None

    # Identify stuck tasks (below target)
    stuck_tasks = {name: data for name, data in all_tasks.items()
                   if data["best_speedup"] < TARGET_SPEEDUP}

    if not stuck_tasks:
        return None

    # Classify each stuck task
    infeasible_tasks = []
    task_hints = {}

    for task_name, data in stuck_tasks.items():
        cluster = classify_failure(task_name, data)

        if cluster == "infeasible":
            infeasible_tasks.append(task_name)

        # Find similar passing task
        similar = find_similar_passing_task(task_name, data, passing_tasks)

        # Find untried strategies
        op_type = data.get("op_type", "other")
        untried = find_untried_strategies(data, op_type)

        # Generate hint
        hint = generate_hint(task_name, data, cluster, similar, untried)

        task_hints[task_name] = {
            "failure_cluster": cluster,
            "best_prior_speedup": round(data["best_speedup"], 3),
            "total_prior_iterations": data["total_iterations"],
            "batches_seen": data["batches_seen"],
            "hint": hint,
        }

        if similar:
            task_hints[task_name]["similar_passing_task"] = similar["task_name"]
            task_hints[task_name]["similar_task_speedup"] = similar["speedup"]
            task_hints[task_name]["similar_task_strategy"] = similar["strategy"]
            if similar.get("kernel_path"):
                task_hints[task_name]["similar_task_kernel_path"] = similar["kernel_path"]

        if untried:
            task_hints[task_name]["untried_strategies"] = untried[:3]

    # Determine escalation tier from manifest
    batches = manifest.get("batches", [])
    current_tier = "plateau"
    if batches:
        last_batch = batches[-1]
        prior_tier = last_batch.get("escalation_tier", "normal")
        if prior_tier in ("plateau", "breakthrough"):
            current_tier = "breakthrough"

    return {
        "escalation_tier": current_tier,
        "generated_at": datetime.now().isoformat(),
        "infeasible_tasks": sorted(infeasible_tasks),
        "cluster_summary": {
            "close_to_target": sum(1 for t in task_hints.values() if t["failure_cluster"] == "close_to_target"),
            "perf_ceiling": sum(1 for t in task_hints.values() if t["failure_cluster"] == "perf_ceiling"),
            "correctness_stuck": sum(1 for t in task_hints.values() if t["failure_cluster"] == "correctness_stuck"),
            "compile_stuck": sum(1 for t in task_hints.values() if t["failure_cluster"] == "compile_stuck"),
            "infeasible": len(infeasible_tasks),
        },
        "tasks": task_hints,
    }


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
        print(f"Error: batch_index must be >= 1 (batch 0 has no prior data to analyze)")
        sys.exit(1)

    # Load chain manifest
    manifest = load_chain_manifest(chain_dir)

    # Build breakthrough hints
    hints = build_breakthrough_hints(manifest, batch_index)

    if not hints:
        print(f"No stuck tasks found from prior batches (0..{batch_index-1})")
        sys.exit(0)

    # Write to session directory
    base = get_output_base()
    session_dir = base / session_id
    if not session_dir.exists():
        print(f"Error: session directory not found: {session_dir}")
        sys.exit(1)

    output_file = session_dir / "breakthrough_hints.json"
    output_file.write_text(json.dumps(hints, indent=2))

    # Print summary
    summary = hints["cluster_summary"]
    total = sum(summary.values())
    print(f"Breakthrough analysis: {total} stuck tasks")
    print(f"  close_to_target:    {summary['close_to_target']}")
    print(f"  perf_ceiling:       {summary['perf_ceiling']}")
    print(f"  correctness_stuck:  {summary['correctness_stuck']}")
    print(f"  compile_stuck:      {summary['compile_stuck']}")
    print(f"  infeasible:         {summary['infeasible']}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
