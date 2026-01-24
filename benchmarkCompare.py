#!/usr/bin/env python
"""
Compare Claude Code kernel benchmark results with RL-trained model results.

This tool generates comparison reports to evaluate Claude Code's kernel
optimization performance against RL fine-tuned models.

Usage:
    python benchmarkCompare.py \
        --claude-dir ~/.inference/claude_code_output/my_session/ \
        --rl-dir ~/.inference/output/my_tag.a_001_01/ \
        --output comparison_report.json

Output includes:
    - Aggregate metrics (success rate, avg speedup)
    - Per-task comparison (which system performed better)
    - Head-to-head summary (wins/losses/ties)
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional
import logging

# Try to use project logger, fall back to standard logging
try:
    from logger import logger
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)


def load_results(result_dir: Path) -> dict:
    """
    Load all eval results from a result directory.

    Handles both Claude Code output format and RL training output format:
    - Claude Code: {session_id}/{task_name}/iteration_XX_eval.json
    - RL: {input_tag}/{task_name}/**/*_generated_eval.json

    Returns:
        Dict mapping task_name -> list of iteration results
    """
    results = {}

    if not result_dir.exists():
        logger.warning(f"Result directory not found: {result_dir}")
        return results

    for task_dir in result_dir.iterdir():
        if not task_dir.is_dir():
            continue
        if task_dir.name.startswith("_") or task_dir.name == "summary.json":
            continue

        task_name = task_dir.name
        iterations = []

        # Try Claude Code format: iteration_XX_eval.json
        eval_files = list(task_dir.glob("iteration_*_eval.json"))

        # Try RL format: **/*_generated_eval.json
        if not eval_files:
            eval_files = list(task_dir.glob("**/*_eval.json"))

        for eval_file in sorted(eval_files):
            try:
                with open(eval_file) as f:
                    result = json.load(f)
                    iterations.append(result)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load {eval_file}: {e}")

        if iterations:
            results[task_name] = iterations

    return results


def compute_metrics(results: dict) -> dict:
    """
    Compute aggregate metrics from results.

    Returns:
        Dict with:
        - total_tasks: Number of unique tasks
        - compiled_rate: Fraction that compiled at least once
        - success_rate: Fraction with at least one correct result
        - avg_speedup: Average best speedup across successful tasks
        - max_speedup: Maximum speedup achieved
        - avg_iterations_to_success: Average iterations needed for first success
    """
    total_tasks = len(results)
    if total_tasks == 0:
        return {
            "total_tasks": 0,
            "compiled_rate": 0,
            "success_rate": 0,
            "avg_speedup": 0,
            "max_speedup": 0,
            "avg_iterations_to_success": 0
        }

    compiled_tasks = 0
    correct_tasks = 0
    speedups = []
    iterations_to_success = []

    for task_name, iterations in results.items():
        task_compiled = False
        task_correct = False
        best_speedup = 0
        first_success_iter = None

        for i, result in enumerate(iterations):
            if result.get("compiled", False):
                task_compiled = True
                if result.get("correctness", False):
                    if not task_correct:
                        first_success_iter = i
                    task_correct = True
                    speedup = result.get("speedup", 0)
                    if speedup > best_speedup:
                        best_speedup = speedup

        if task_compiled:
            compiled_tasks += 1
        if task_correct:
            correct_tasks += 1
            speedups.append(best_speedup)
            if first_success_iter is not None:
                iterations_to_success.append(first_success_iter + 1)

    return {
        "total_tasks": total_tasks,
        "compiled_rate": compiled_tasks / total_tasks,
        "success_rate": correct_tasks / total_tasks,
        "avg_speedup": sum(speedups) / len(speedups) if speedups else 0,
        "max_speedup": max(speedups) if speedups else 0,
        "avg_iterations_to_success": sum(iterations_to_success) / len(iterations_to_success) if iterations_to_success else 0
    }


def compare(claude_dir: Path, rl_dir: Path) -> dict:
    """
    Compare Claude Code results with RL model results.

    Returns:
        Dict with:
        - claude_metrics: Aggregate metrics for Claude Code
        - rl_metrics: Aggregate metrics for RL model
        - task_comparison: Per-task comparison
        - summary: Overall head-to-head summary
    """
    claude_results = load_results(claude_dir)
    rl_results = load_results(rl_dir)

    claude_metrics = compute_metrics(claude_results)
    rl_metrics = compute_metrics(rl_results)

    # Per-task comparison
    all_tasks = set(claude_results.keys()) | set(rl_results.keys())
    task_comparison = {}

    for task in sorted(all_tasks):
        claude_best = 0
        rl_best = 0
        claude_correct = False
        rl_correct = False

        if task in claude_results:
            for r in claude_results[task]:
                if r.get("correctness"):
                    claude_correct = True
                    speedup = r.get("speedup", 0)
                    if speedup > claude_best:
                        claude_best = speedup

        if task in rl_results:
            for r in rl_results[task]:
                if r.get("correctness"):
                    rl_correct = True
                    speedup = r.get("speedup", 0)
                    if speedup > rl_best:
                        rl_best = speedup

        # Determine winner
        if claude_correct and not rl_correct:
            winner = "claude"
        elif rl_correct and not claude_correct:
            winner = "rl"
        elif claude_best > rl_best:
            winner = "claude"
        elif rl_best > claude_best:
            winner = "rl"
        else:
            winner = "tie"

        task_comparison[task] = {
            "claude_speedup": claude_best,
            "claude_correct": claude_correct,
            "rl_speedup": rl_best,
            "rl_correct": rl_correct,
            "winner": winner,
            "speedup_diff": claude_best - rl_best
        }

    # Summary
    claude_wins = sum(1 for t in task_comparison.values() if t["winner"] == "claude")
    rl_wins = sum(1 for t in task_comparison.values() if t["winner"] == "rl")
    ties = sum(1 for t in task_comparison.values() if t["winner"] == "tie")

    return {
        "claude_metrics": claude_metrics,
        "rl_metrics": rl_metrics,
        "task_comparison": task_comparison,
        "summary": {
            "total_tasks_compared": len(task_comparison),
            "claude_wins": claude_wins,
            "rl_wins": rl_wins,
            "ties": ties,
            "claude_win_rate": claude_wins / len(task_comparison) if task_comparison else 0
        }
    }


def print_report(report: dict):
    """Print a formatted comparison report to stdout."""
    print("\n" + "=" * 60)
    print("KERNEL BENCHMARK COMPARISON REPORT")
    print("=" * 60)

    print("\n--- Aggregate Metrics ---\n")
    print(f"{'Metric':<30} {'Claude Code':>15} {'RL Model':>15}")
    print("-" * 60)

    cm = report["claude_metrics"]
    rm = report["rl_metrics"]

    print(f"{'Total Tasks':<30} {cm['total_tasks']:>15} {rm['total_tasks']:>15}")
    print(f"{'Compiled Rate':<30} {cm['compiled_rate']:>14.1%} {rm['compiled_rate']:>14.1%}")
    print(f"{'Success Rate':<30} {cm['success_rate']:>14.1%} {rm['success_rate']:>14.1%}")
    print(f"{'Avg Speedup (correct only)':<30} {cm['avg_speedup']:>14.2f}x {rm['avg_speedup']:>14.2f}x")
    print(f"{'Max Speedup':<30} {cm['max_speedup']:>14.2f}x {rm['max_speedup']:>14.2f}x")
    print(f"{'Avg Iterations to Success':<30} {cm['avg_iterations_to_success']:>15.1f} {rm['avg_iterations_to_success']:>15.1f}")

    print("\n--- Head-to-Head Summary ---\n")
    s = report["summary"]
    print(f"Tasks compared: {s['total_tasks_compared']}")
    print(f"Claude Code wins: {s['claude_wins']} ({s['claude_wins']/max(1, s['total_tasks_compared']):.1%})")
    print(f"RL Model wins:    {s['rl_wins']} ({s['rl_wins']/max(1, s['total_tasks_compared']):.1%})")
    print(f"Ties:             {s['ties']}")

    # Show notable differences
    print("\n--- Notable Results ---\n")
    tc = report["task_comparison"]

    # Tasks where Claude wins significantly
    claude_big_wins = [(t, d) for t, d in tc.items()
                       if d["winner"] == "claude" and d["speedup_diff"] > 0.2]
    if claude_big_wins:
        print("Claude Code significantly better (>0.2x speedup advantage):")
        for task, data in sorted(claude_big_wins, key=lambda x: -x[1]["speedup_diff"])[:5]:
            print(f"  {task}: {data['claude_speedup']:.2f}x vs {data['rl_speedup']:.2f}x "
                  f"(+{data['speedup_diff']:.2f}x)")

    # Tasks where RL wins significantly
    rl_big_wins = [(t, d) for t, d in tc.items()
                   if d["winner"] == "rl" and d["speedup_diff"] < -0.2]
    if rl_big_wins:
        print("\nRL Model significantly better (>0.2x speedup advantage):")
        for task, data in sorted(rl_big_wins, key=lambda x: x[1]["speedup_diff"])[:5]:
            print(f"  {task}: {data['rl_speedup']:.2f}x vs {data['claude_speedup']:.2f}x "
                  f"(+{-data['speedup_diff']:.2f}x)")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Claude Code vs RL model kernel benchmark results"
    )
    parser.add_argument(
        "--claude-dir",
        required=True,
        help="Claude Code results directory (e.g., ~/.inference/claude_code_output/my_session/)"
    )
    parser.add_argument(
        "--rl-dir",
        required=True,
        help="RL model results directory (e.g., ~/.inference/output/my_tag.a_001_01/)"
    )
    parser.add_argument(
        "--output",
        default="comparison_report.json",
        help="Output JSON report file path"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output JSON, no printed report"
    )
    args = parser.parse_args()

    claude_dir = Path(os.path.expanduser(args.claude_dir))
    rl_dir = Path(os.path.expanduser(args.rl_dir))

    logger.info(f"Comparing results:")
    logger.info(f"  Claude Code: {claude_dir}")
    logger.info(f"  RL Model:    {rl_dir}")

    report = compare(claude_dir, rl_dir)

    # Save JSON report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to: {output_path}")

    # Print formatted report
    if not args.quiet:
        print_report(report)


if __name__ == "__main__":
    main()
