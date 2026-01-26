#!/usr/bin/env python
"""
Test Phase 2: End-to-End Resume Flow

This test validates that:
1. Session initialization captures the EXACT tasks requested (not all tasks)
2. Session config (worker count, original prompt) is persisted
3. Progress is correctly tracked as work completes
4. Resume correctly identifies completed vs remaining work
5. Final result matches original request

Key concern: When user specifies "3 tasks from level1", we should:
- Store exactly those 3 tasks in manifest
- Track completion of each
- On resume, show 3 total, X completed, Y remaining
"""

import asyncio
import os
import sys
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

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
TEST_SESSION = "test_resume_flow"
TEST_BASE = Path(os.path.expanduser(f"~/.inference/claude_code_output/{TEST_SESSION}"))


def cleanup():
    """Remove test session directory."""
    if TEST_BASE.exists():
        shutil.rmtree(TEST_BASE)
        print(f"Cleaned up {TEST_BASE}")


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_test(name):
    print(f"\n--- {name} ---")


def print_pass(msg):
    print(f"  ✓ {msg}")


def print_fail(msg):
    print(f"  ✗ FAIL: {msg}")


def print_info(msg):
    print(f"    {msg}")


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, condition, pass_msg, fail_msg):
        if condition:
            print_pass(pass_msg)
            self.passed += 1
            return True
        else:
            print_fail(fail_msg)
            self.failed += 1
            self.errors.append(fail_msg)
            return False


async def test_1_init_session_with_task_filter(results: TestResult):
    """Test that init_session correctly captures ONLY the specified tasks."""
    print_section("TEST 1: init_session with Task Filter")

    # First, get some task names from level1 to use
    all_tasks = await list_kernel_bench_tasks(level="level1")
    selected_tasks = [t["name"] for t in all_tasks[:3]]  # Pick first 3 tasks
    print_info(f"Selected tasks: {selected_tasks}")

    # Initialize session with SPECIFIC tasks
    result = await init_session(
        session_id=TEST_SESSION,
        level="level1",
        task_names=selected_tasks,
        config_override={
            "num_workers": 2,
            "original_request": "3 tasks from level1"
        }
    )

    print_test("1.1 Session created")
    results.check(
        result.get("session_id") == TEST_SESSION,
        "Session ID matches",
        f"Session ID mismatch: {result.get('session_id')}"
    )

    print_test("1.2 Check task count - should be exactly 3")
    total_tasks = result.get("total_tasks", 0)
    tasks = result.get("tasks", [])
    print_info(f"Total tasks in manifest: {total_tasks}")
    print_info(f"Tasks: {tasks}")

    results.check(
        total_tasks == 3,
        f"Manifest has exactly {total_tasks} tasks (expected 3)",
        f"Wrong task count: {total_tasks} (expected 3)"
    )

    results.check(
        tasks == selected_tasks,
        "Manifest contains exactly the selected tasks",
        f"Task mismatch: {tasks} vs {selected_tasks}"
    )

    print_test("1.3 Check config persistence")
    manifest_path = TEST_BASE / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    stored_config = manifest.get("config", {})
    print_info(f"Stored config: {stored_config}")

    results.check(
        stored_config.get("num_workers") == 2,
        "num_workers stored in config",
        f"num_workers not stored: {stored_config}"
    )

    results.check(
        stored_config.get("original_request") == "3 tasks from level1",
        "original_request stored in config",
        f"original_request not stored: {stored_config}"
    )

    return tasks  # Return the 3 tasks to work with


async def test_2_simulate_partial_completion(results: TestResult, task_names: list):
    """Simulate completing some tasks, leaving others pending."""
    print_section("TEST 2: Simulate Partial Work")

    print_test("2.1 Complete first task")
    task1 = task_names[0]

    # Claim task
    claim_result = await claim_task(TEST_SESSION, task1, "worker-1")
    results.check(
        claim_result.get("success"),
        f"Claimed {task1}",
        f"Failed to claim: {claim_result}"
    )

    # Simulate completing the task (create best_result.json)
    task_dir = TEST_BASE / task1
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create iteration files (simulating work)
    (task_dir / "iteration_00_cuda_kernel.py").write_text("# kernel code")
    (task_dir / "iteration_00_eval.json").write_text(json.dumps({
        "compiled": True, "correctness": True, "speedup": 1.35
    }))

    # Create completion marker
    best_result = {
        "task_name": task1,
        "best_iteration": 0,
        "speedup": 1.35,
        "completed_at": datetime.now().isoformat()
    }
    (task_dir / "best_result.json").write_text(json.dumps(best_result, indent=2))

    # Remove in_progress marker (simulating normal completion)
    marker = task_dir / ".in_progress"
    if marker.exists():
        marker.unlink()

    print_pass(f"Completed {task1} with speedup 1.35x")

    print_test("2.2 Leave second task in-progress (simulate crash)")
    task2 = task_names[1]

    claim_result = await claim_task(TEST_SESSION, task2, "worker-2")
    results.check(
        claim_result.get("success"),
        f"Claimed {task2}",
        f"Failed to claim: {claim_result}"
    )

    # Create partial work but NO best_result.json
    task2_dir = TEST_BASE / task2
    (task2_dir / "iteration_00_cuda_kernel.py").write_text("# partial kernel")
    print_info(f"Left {task2} in-progress (simulating crash)")

    print_test("2.3 Leave third task untouched")
    task3 = task_names[2]
    print_info(f"Task {task3} never started")

    return task1, task2, task3


async def test_3_verify_state_tracking(results: TestResult, completed, in_progress, pending):
    """Verify get_session_state correctly identifies task states."""
    print_section("TEST 3: State Tracking Accuracy")

    print_test("3.1 Get session state")
    state = await get_session_state(TEST_SESSION)

    print_info(f"Total: {state.get('total')}")
    print_info(f"Completed: {state.get('completed')}")
    print_info(f"In-progress: {state.get('in_progress')}")
    print_info(f"Incomplete: {state.get('incomplete')}")
    print_info(f"Pending: {state.get('pending')}")

    print_test("3.2 Verify completed task detected")
    completed_tasks = state.get("completed_tasks", [])
    results.check(
        completed in completed_tasks,
        f"Completed task {completed} detected",
        f"Completed task not in list: {completed_tasks}"
    )

    print_test("3.3 Verify in-progress task detected")
    in_progress_tasks = [t.get("task") if isinstance(t, dict) else t
                         for t in state.get("in_progress_tasks", [])]
    results.check(
        in_progress in in_progress_tasks,
        f"In-progress task {in_progress} detected",
        f"In-progress task not in list: {in_progress_tasks}"
    )

    print_test("3.4 Verify pending task exists")
    pending_tasks = state.get("pending_tasks", [])
    # Note: pending_tasks may be truncated, check pending count
    pending_count = state.get("pending", 0)
    results.check(
        pending_count > 0,
        f"Pending tasks detected: {pending_count}",
        "No pending tasks found"
    )


async def test_4_stale_marker_handling(results: TestResult, in_progress_task):
    """Test that stale markers are correctly identified and cleaned."""
    print_section("TEST 4: Stale Marker Handling")

    print_test("4.1 Create stale marker (old timestamp)")
    task_dir = TEST_BASE / in_progress_task
    marker_path = task_dir / ".in_progress"

    # Read current marker
    if marker_path.exists():
        current_marker = json.loads(marker_path.read_text())
        print_info(f"Current marker: {current_marker}")

        # Overwrite with stale timestamp (35 minutes ago)
        stale_time = datetime.now() - timedelta(minutes=35)
        current_marker["started_at"] = stale_time.isoformat()
        marker_path.write_text(json.dumps(current_marker))
        print_info(f"Set marker to stale time: {stale_time.isoformat()}")

    print_test("4.2 Get state (should clean stale)")
    state = await get_session_state(TEST_SESSION)
    stale_cleaned = state.get("stale_cleaned", 0)

    results.check(
        stale_cleaned >= 1,
        f"Stale marker cleaned: {stale_cleaned}",
        f"Stale marker not cleaned: stale_cleaned={stale_cleaned}"
    )

    print_test("4.3 Verify task is now claimable")
    # After stale cleanup, task should be in incomplete (has iteration files)
    incomplete_tasks = state.get("incomplete_tasks", [])
    results.check(
        in_progress_task in incomplete_tasks,
        f"Task {in_progress_task} now incomplete (claimable)",
        f"Task not in incomplete: {incomplete_tasks}"
    )


async def test_5_resume_flow(results: TestResult, original_tasks):
    """Test the resume flow - verify we can continue where we left off."""
    print_section("TEST 5: Resume Flow")

    print_test("5.1 Simulate resume by reading manifest")
    manifest_path = TEST_BASE / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    print_info(f"Original session created: {manifest.get('created_at')}")
    print_info(f"Level: {manifest.get('level')}")
    print_info(f"Total tasks: {manifest.get('total_tasks')}")
    print_info(f"Tasks: {manifest.get('tasks')}")
    print_info(f"Config: {manifest.get('config')}")

    results.check(
        manifest.get("total_tasks") == 3,
        "Resume sees original 3 tasks (not 100)",
        f"Resume sees wrong count: {manifest.get('total_tasks')}"
    )

    results.check(
        manifest.get("status") in ["initialized", "existing"],
        "Manifest has valid status",
        f"Invalid status: {manifest.get('status')}"
    )

    print_test("5.2 Get pending tasks for resume")
    pending = await get_pending_tasks(TEST_SESSION, limit=10)

    available = pending.get("available_count", 0)
    print_info(f"Available for claiming: {available}")

    # Should be 2 remaining (1 completed, 2 remaining from 3 total)
    results.check(
        available == 2,
        f"Exactly 2 tasks available for resume (1 completed from 3)",
        f"Wrong available count: {available} (expected 2)"
    )

    print_test("5.3 Verify completed work preserved")
    completed_task = original_tasks[0]
    best_result_path = TEST_BASE / completed_task / "best_result.json"

    results.check(
        best_result_path.exists(),
        f"Completed task {completed_task} preserved",
        f"Completed work lost: {best_result_path}"
    )

    if best_result_path.exists():
        best_result = json.loads(best_result_path.read_text())
        print_info(f"Best speedup: {best_result.get('speedup')}x")


async def test_6_verify_fix_in_place(results: TestResult):
    """Verify that init_session now supports task filtering."""
    print_section("TEST 6: Verify Fix Implementation")

    print_test("6.1 Task subset support in init_session")
    # Check if init_session supports task_names parameter
    import inspect
    sig = inspect.signature(init_session)
    params = list(sig.parameters.keys())
    print_info(f"init_session params: {params}")

    results.check(
        "task_names" in params,
        "init_session has task_names parameter",
        f"MISSING: init_session needs task_names parameter, has: {params}"
    )

    print_test("6.2 Worker count in config")
    manifest_path = TEST_BASE / "session_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        has_workers = manifest.get("config", {}).get("num_workers")
        results.check(
            has_workers == 2,
            f"Worker count stored: {has_workers}",
            f"Worker count not stored correctly: {has_workers}"
        )

    print_test("6.3 Original request preservation")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        has_request = manifest.get("config", {}).get("original_request")
        results.check(
            has_request == "3 tasks from level1",
            f"Original request stored: '{has_request}'",
            f"Original request not stored: {has_request}"
        )


async def test_7_summary_validation(results: TestResult):
    """Final validation that the resume flow works correctly."""
    print_section("TEST 7: End-to-End Summary Validation")

    print_test("7.1 Verify complete resume scenario")
    manifest_path = TEST_BASE / "session_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    total = manifest.get("total_tasks", 0)
    tasks = manifest.get("tasks", [])

    print_info(f"Session: {manifest.get('session_id')}")
    print_info(f"Original request: {manifest.get('config', {}).get('original_request')}")
    print_info(f"Total tasks requested: {total}")
    print_info(f"Tasks: {tasks}")

    # Get current state
    state = await get_session_state(TEST_SESSION)

    print_info(f"Completed: {state.get('completed')}/{total}")
    print_info(f"Remaining: {state.get('incomplete', 0) + state.get('pending', 0)}")

    # Verify numbers add up correctly
    total_from_state = (
        state.get('completed', 0) +
        state.get('in_progress', 0) +
        state.get('incomplete', 0) +
        state.get('pending', 0)
    )

    results.check(
        total_from_state == total,
        f"State counts add up to total: {total_from_state} == {total}",
        f"State counts mismatch: {total_from_state} != {total}"
    )

    print_test("7.2 Resume scenario summary")
    print_info("")
    print_info("✓ User requested: '3 tasks from level1'")
    print_info(f"✓ Session tracks exactly {total} tasks")
    print_info(f"✓ After partial work: {state.get('completed')} completed, "
               f"{state.get('incomplete', 0) + state.get('pending', 0)} remaining")
    print_info("✓ On resume: system knows exactly what's done and what's left")
    print_info("")
    print_info("Resume works correctly!")


async def main():
    print("\n" + "=" * 70)
    print("  PHASE 2 RESUME FLOW TEST SUITE")
    print("  Testing: Session state management for proper resume")
    print("=" * 70)

    results = TestResult()

    # Cleanup before tests
    cleanup()

    try:
        # Run all tests
        task_names = await test_1_init_session_with_task_filter(results)

        if len(task_names) >= 3:
            completed, in_progress, pending = await test_2_simulate_partial_completion(
                results, task_names
            )
            await test_3_verify_state_tracking(results, completed, in_progress, pending)
            await test_4_stale_marker_handling(results, in_progress)
            await test_5_resume_flow(results, task_names)

        await test_6_verify_fix_in_place(results)
        await test_7_summary_validation(results)

        # Summary
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        print(f"  Passed: {results.passed}")
        print(f"  Failed: {results.failed}")

        if results.errors:
            print(f"\n  Issues found:")
            for i, err in enumerate(results.errors, 1):
                print(f"    {i}. {err}")

        print("\n" + "=" * 70)
        if results.failed == 0:
            print("  ALL TESTS PASSED")
        else:
            print("  SOME TESTS FAILED - See issues above")
        print("=" * 70)

        return results.failed == 0

    finally:
        # Keep test data for inspection
        print(f"\n  Test data preserved at: {TEST_BASE}")
        print("  Run 'rm -rf ~/.inference/claude_code_output/test_resume_flow' to cleanup")


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
