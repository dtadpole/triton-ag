#!/usr/bin/env python
"""
Phase 2 Comprehensive Test Suite

Tests all unit-testable items from Phase 2.md Section 6.1:
- T5: Atomic Claim - Two workers claim same task, only one succeeds
- T6: Stale Cleanup (time) - Create 35-min old marker, verify auto-cleanup
- T7: PID Cleanup - Create marker with dead PID, verify cleanup
- T9: Resume - Kill mid-session, resume, verify continuation
- T10: Parallel Workers - 4 workers, 20 tasks, no duplicates claimed

Also tests:
- Task filtering in init_session
- Config persistence
- State tracking accuracy
"""

import asyncio
import os
import sys
import json
import shutil
import socket
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

from claudeCodeKernelBenchServer import (
    init_session,
    claim_task,
    release_task,
    get_session_state,
    get_pending_tasks,
    list_kernel_bench_tasks,
    config,
)

# Test constants
TEST_BASE = Path(os.path.expanduser("~/.inference/claude_code_output"))


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, pass_msg, fail_msg):
        if condition:
            print(f"  ✓ {pass_msg}")
            self.passed += 1
            return True
        else:
            print(f"  ✗ FAIL: {fail_msg}")
            self.failed += 1
            self.errors.append(fail_msg)
            return False


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_test(name):
    print(f"\n--- {name} ---")


def cleanup_session(session_id):
    """Remove test session directory."""
    session_dir = TEST_BASE / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)


# =============================================================================
# T5: Atomic Claim
# =============================================================================

async def test_T5_atomic_claim(results: TestResult):
    """T5: Two workers claim same task, only one succeeds."""
    print_section("T5: Atomic Claim")

    session_id = "test_T5_atomic"
    cleanup_session(session_id)

    try:
        # Initialize session with a few tasks
        all_tasks = await list_kernel_bench_tasks(level="level1")
        task_names = [t["name"] for t in all_tasks[:5]]

        await init_session(
            session_id=session_id,
            level="level1",
            task_names=task_names
        )

        task_to_claim = task_names[0]

        print_test("T5.1: First worker claims task")
        r1 = await claim_task(session_id, task_to_claim, "worker-1")
        results.check(
            r1.get("success") is True,
            f"Worker-1 successfully claimed {task_to_claim}",
            f"Worker-1 failed to claim: {r1}"
        )

        print_test("T5.2: Second worker tries same task")
        r2 = await claim_task(session_id, task_to_claim, "worker-2")
        results.check(
            r2.get("success") is False,
            "Worker-2 correctly rejected (task already claimed)",
            f"Worker-2 should have been rejected: {r2}"
        )

        results.check(
            r2.get("reason") == "already_claimed",
            f"Rejection reason is 'already_claimed'",
            f"Wrong rejection reason: {r2.get('reason')}"
        )

        print_test("T5.3: Workers claim different tasks")
        r3 = await claim_task(session_id, task_names[1], "worker-2")
        results.check(
            r3.get("success") is True,
            f"Worker-2 successfully claimed different task: {task_names[1]}",
            f"Worker-2 failed to claim different task: {r3}"
        )

    finally:
        cleanup_session(session_id)


# =============================================================================
# T6: Stale Cleanup (Time-based)
# =============================================================================

async def test_T6_stale_cleanup_time(results: TestResult):
    """T6: Create 35-min old marker, verify auto-cleanup."""
    print_section("T6: Stale Cleanup (Time-based)")

    session_id = "test_T6_stale_time"
    cleanup_session(session_id)

    try:
        # Initialize session
        all_tasks = await list_kernel_bench_tasks(level="level1")
        task_names = [t["name"] for t in all_tasks[:3]]

        await init_session(
            session_id=session_id,
            level="level1",
            task_names=task_names
        )

        task_name = task_names[0]

        print_test("T6.1: Claim task normally")
        r = await claim_task(session_id, task_name, "worker-1")
        results.check(r.get("success"), "Task claimed", f"Claim failed: {r}")

        print_test("T6.2: Modify marker to be 35 minutes old")
        task_dir = TEST_BASE / session_id / task_name
        marker_path = task_dir / ".in_progress"

        marker_data = json.loads(marker_path.read_text())
        old_time = datetime.now() - timedelta(minutes=35)
        marker_data["started_at"] = old_time.isoformat()
        marker_path.write_text(json.dumps(marker_data))

        print(f"    Set marker time to: {old_time.isoformat()}")

        print_test("T6.3: Get session state (triggers stale cleanup)")
        state = await get_session_state(session_id)

        results.check(
            state.get("stale_cleaned", 0) >= 1,
            f"Stale marker cleaned: {state.get('stale_cleaned')}",
            f"Stale marker NOT cleaned: stale_cleaned={state.get('stale_cleaned')}"
        )

        print_test("T6.4: Verify marker file removed")
        results.check(
            not marker_path.exists(),
            "Marker file successfully removed",
            "Marker file still exists!"
        )

        print_test("T6.5: Task is now claimable again")
        r2 = await claim_task(session_id, task_name, "worker-2")
        results.check(
            r2.get("success") is True,
            "Task can be re-claimed after stale cleanup",
            f"Task cannot be re-claimed: {r2}"
        )

    finally:
        cleanup_session(session_id)


# =============================================================================
# T7: PID Cleanup (Dead process)
# =============================================================================

async def test_T7_pid_cleanup(results: TestResult):
    """T7: Create marker with dead PID, verify cleanup."""
    print_section("T7: PID Cleanup (Dead Process)")

    session_id = "test_T7_dead_pid"
    cleanup_session(session_id)

    try:
        # Initialize session
        all_tasks = await list_kernel_bench_tasks(level="level1")
        task_names = [t["name"] for t in all_tasks[:3]]

        await init_session(
            session_id=session_id,
            level="level1",
            task_names=task_names
        )

        task_name = task_names[0]
        task_dir = TEST_BASE / session_id / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        print_test("T7.1: Create marker with dead PID")
        # Use a PID that definitely doesn't exist (very high number)
        dead_pid = 99999999

        marker_data = {
            "worker": "dead-worker",
            "started_at": datetime.now().isoformat(),  # Recent time
            "pid": dead_pid,
            "hostname": socket.gethostname()  # Same host so PID check happens
        }

        marker_path = task_dir / ".in_progress"
        marker_path.write_text(json.dumps(marker_data))

        print(f"    Created marker with PID: {dead_pid}")

        # Verify the PID doesn't exist
        try:
            os.kill(dead_pid, 0)
            print("    WARNING: PID exists (unexpected)")
        except OSError:
            print("    Confirmed: PID does not exist")

        print_test("T7.2: Get session state (triggers PID-based cleanup)")
        state = await get_session_state(session_id)

        results.check(
            state.get("stale_cleaned", 0) >= 1,
            f"Dead PID marker cleaned: {state.get('stale_cleaned')}",
            f"Dead PID marker NOT cleaned: stale_cleaned={state.get('stale_cleaned')}"
        )

        print_test("T7.3: Verify marker file removed")
        results.check(
            not marker_path.exists(),
            "Marker file successfully removed",
            "Marker file still exists!"
        )

        print_test("T7.4: Task is now claimable")
        r = await claim_task(session_id, task_name, "worker-new")
        results.check(
            r.get("success") is True,
            "Task can be claimed after dead PID cleanup",
            f"Task cannot be claimed: {r}"
        )

    finally:
        cleanup_session(session_id)


# =============================================================================
# T9: Resume (State Management)
# =============================================================================

async def test_T9_resume(results: TestResult):
    """T9: Simulate crash, resume, verify continuation."""
    print_section("T9: Resume (Crash Recovery)")

    session_id = "test_T9_resume"
    cleanup_session(session_id)

    try:
        # Initialize session with specific tasks
        all_tasks = await list_kernel_bench_tasks(level="level1")
        task_names = [t["name"] for t in all_tasks[:5]]

        print_test("T9.1: Initialize session with 5 tasks")
        await init_session(
            session_id=session_id,
            level="level1",
            task_names=task_names,
            config_override={"num_workers": 2, "original_request": "5 specific tasks"}
        )

        print(f"    Tasks: {task_names}")

        print_test("T9.2: Complete 2 tasks")
        for i in range(2):
            task = task_names[i]
            await claim_task(session_id, task, f"worker-{i}")

            # Create completion marker
            task_dir = TEST_BASE / session_id / task
            (task_dir / "iteration_00_cuda_kernel.py").write_text("# kernel")
            (task_dir / "best_result.json").write_text(json.dumps({
                "speedup": 1.3 + i * 0.1,
                "completed_at": datetime.now().isoformat()
            }))

            # Remove in-progress marker
            marker = task_dir / ".in_progress"
            if marker.exists():
                marker.unlink()

            print(f"    Completed: {task}")

        print_test("T9.3: Leave 1 task in-progress (simulate crash)")
        crash_task = task_names[2]
        await claim_task(session_id, crash_task, "crashed-worker")
        print(f"    In-progress (crashed): {crash_task}")

        print_test("T9.4: Simulate resume - check session state")
        state = await get_session_state(session_id)

        results.check(
            state.get("total") == 5,
            f"Resume sees correct total: {state.get('total')}",
            f"Wrong total: {state.get('total')}"
        )

        results.check(
            state.get("completed") == 2,
            f"Resume sees 2 completed tasks",
            f"Wrong completed count: {state.get('completed')}"
        )

        results.check(
            state.get("in_progress") == 1,
            f"Resume sees 1 in-progress task",
            f"Wrong in-progress count: {state.get('in_progress')}"
        )

        results.check(
            state.get("pending") == 2,
            f"Resume sees 2 pending tasks",
            f"Wrong pending count: {state.get('pending')}"
        )

        print_test("T9.5: Get pending tasks for resume")
        pending = await get_pending_tasks(session_id, limit=10)

        # Should be 2 pending (the in-progress one is not available until stale)
        results.check(
            pending.get("available_count") == 2,
            f"2 tasks available for new workers",
            f"Wrong available count: {pending.get('available_count')}"
        )

        print_test("T9.6: Verify completed work preserved")
        for i in range(2):
            task = task_names[i]
            best_result = TEST_BASE / session_id / task / "best_result.json"
            results.check(
                best_result.exists(),
                f"Completed task {task} preserved",
                f"Completed task {task} lost!"
            )

    finally:
        cleanup_session(session_id)


# =============================================================================
# T10: Parallel Workers (Concurrency)
# =============================================================================

async def test_T10_parallel_workers(results: TestResult):
    """T10: 4 workers, 20 tasks, no duplicates claimed."""
    print_section("T10: Parallel Workers (Concurrency)")

    session_id = "test_T10_parallel"
    cleanup_session(session_id)

    try:
        # Initialize session with 20 tasks
        all_tasks = await list_kernel_bench_tasks(level="level1")
        task_names = [t["name"] for t in all_tasks[:20]]

        await init_session(
            session_id=session_id,
            level="level1",
            task_names=task_names
        )

        print_test("T10.1: 4 workers race to claim 20 tasks")

        claimed_by = {}  # task -> worker
        claim_results = []

        async def worker_loop(worker_id: str, max_claims: int = 10):
            """Simulate a worker claiming tasks."""
            claims = []
            for task in task_names:
                if len(claims) >= max_claims:
                    break
                result = await claim_task(session_id, task, worker_id)
                if result.get("success"):
                    claims.append(task)
            return worker_id, claims

        # Run 4 workers concurrently
        worker_tasks = [
            worker_loop(f"worker-{i}", max_claims=10)
            for i in range(4)
        ]

        worker_results = await asyncio.gather(*worker_tasks)

        # Analyze results
        total_claims = 0
        for worker_id, claims in worker_results:
            print(f"    {worker_id}: claimed {len(claims)} tasks")
            total_claims += len(claims)
            for task in claims:
                if task in claimed_by:
                    results.check(
                        False,
                        "",
                        f"DUPLICATE: {task} claimed by both {claimed_by[task]} and {worker_id}"
                    )
                else:
                    claimed_by[task] = worker_id

        print_test("T10.2: Verify no duplicates")
        results.check(
            len(claimed_by) == total_claims,
            f"No duplicates: {len(claimed_by)} unique claims = {total_claims} total claims",
            f"Duplicates detected! {len(claimed_by)} unique != {total_claims} total"
        )

        print_test("T10.3: Verify all 20 tasks claimed")
        results.check(
            len(claimed_by) == 20,
            f"All 20 tasks claimed",
            f"Only {len(claimed_by)} tasks claimed"
        )

        print_test("T10.4: Verify state tracking")
        state = await get_session_state(session_id)
        results.check(
            state.get("in_progress") == 20,
            f"State shows 20 in-progress",
            f"State shows {state.get('in_progress')} in-progress"
        )

    finally:
        cleanup_session(session_id)


# =============================================================================
# Additional: Task Filtering
# =============================================================================

async def test_task_filtering(results: TestResult):
    """Test that init_session correctly filters to specific tasks."""
    print_section("Additional: Task Filtering")

    session_id = "test_task_filter"
    cleanup_session(session_id)

    try:
        # Get some tasks
        all_tasks = await list_kernel_bench_tasks(level="level1")
        selected = [t["name"] for t in all_tasks[10:13]]  # Pick 3 specific tasks

        print_test("Initialize with 3 specific tasks")
        result = await init_session(
            session_id=session_id,
            level="level1",
            task_names=selected,
            config_override={"num_workers": 2}
        )

        results.check(
            result.get("total_tasks") == 3,
            f"Manifest has 3 tasks (not 100)",
            f"Wrong count: {result.get('total_tasks')}"
        )

        results.check(
            result.get("tasks") == selected,
            "Manifest has exactly the selected tasks",
            f"Task mismatch: {result.get('tasks')}"
        )

        print_test("Without task_names, gets all tasks")
        session2 = "test_task_filter_all"
        cleanup_session(session2)

        result2 = await init_session(
            session_id=session2,
            level="level1"
        )

        results.check(
            result2.get("total_tasks") == 100,
            "Without filter: all 100 level1 tasks",
            f"Wrong count: {result2.get('total_tasks')}"
        )

        cleanup_session(session2)

    finally:
        cleanup_session(session_id)


# =============================================================================
# Main
# =============================================================================

async def main():
    print("\n" + "=" * 70)
    print("  PHASE 2 COMPREHENSIVE TEST SUITE")
    print("  Testing: T5, T6, T7, T9, T10 from Phase 2.md Section 6.1")
    print("=" * 70)

    results = TestResult()

    # Run all tests
    await test_T5_atomic_claim(results)
    await test_T6_stale_cleanup_time(results)
    await test_T7_pid_cleanup(results)
    await test_T9_resume(results)
    await test_T10_parallel_workers(results)
    await test_task_filtering(results)

    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")

    if results.errors:
        print(f"\n  Errors:")
        for i, err in enumerate(results.errors, 1):
            print(f"    {i}. {err}")

    print("\n" + "=" * 70)
    if results.failed == 0:
        print("  ALL TESTS PASSED ✓")
    else:
        print(f"  {results.failed} TESTS FAILED")
    print("=" * 70)

    return results.failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
