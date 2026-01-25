"""Test Phase 6.1: Adaptive semaphore initialization."""
import asyncio
import sys
sys.path.insert(0, '/Users/aarontao/Projects/code/triton-ag')

import claudeCodeKernelBenchServer as mcp_server

async def main():
    print("=== Test 6.1: Adaptive Semaphore Initialization ===\n")

    # Test 1: Query /info endpoint directly
    print("Step 1: Query /info endpoint")
    client = mcp_server.get_kbeval_client()
    try:
        info = await client.get_info(provider="local")
        if info is None:
            print("  ✗ Server returned None - /info endpoint may not exist")
            print("  (Ensure kbEvalServer is updated with /info endpoint)")
            return
        print(f"  Server info: {info}")
        num_devices = info.get("num_devices", 1)
        print(f"  ✓ num_devices = {num_devices}")
    except Exception as e:
        print(f"  ✗ Failed to get info: {e}")
        return

    # Test 2: Initialize semaphore via get_eval_semaphore
    print("\nStep 2: Initialize adaptive semaphore")
    semaphore = await mcp_server.get_eval_semaphore(provider="local")
    semaphore_size = mcp_server._semaphore_size

    print(f"  ✓ Semaphore initialized with {semaphore_size} slots")
    if semaphore_size == num_devices:
        print("  ✓ Semaphore matches device count")
        print("\n=== Phase 6.1: PASS ===")
    else:
        print(f"  ✗ Mismatch: semaphore={semaphore_size}, devices={num_devices}")
        print("\n=== Phase 6.1: FAIL ===")

if __name__ == "__main__":
    asyncio.run(main())
