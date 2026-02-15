# Phase 6: Server Reliability

## Problem

The kbEval server (`kbEvalServer.py`) crashes mid-session, breaking active kernel-bench skill sessions. The current recovery mechanism is a shell loop (`while true; do python kbEvalServer.py ...; sleep 5; done`) that only restarts after the process fully exits. This has several gaps:

1. **Doesn't catch "alive but stuck" states** — the process may be running but unresponsive (hung GPU operations, deadlocked locks, exhausted file descriptors).
2. **No health verification after restart** — the loop restarts immediately without confirming the server is actually serving.
3. **No GPU health check** — if the crash was caused by a GPU issue, restarting without checking GPU state may just crash again.
4. **Client-side sessions break** — the kernel-bench skill and its MCP server run on a separate dev server connected via SSH tunnel. When the eval server dies, the client-side Claude agent gets a hard tool error and the session is effectively lost.

## Architecture

```
[Dev Server]                              [Eval Server Machine]
  Claude Code session                       kbEvalServer.py (FastAPI)
  kernel-bench skill                        kbEvalWatchdog.py (new)
  claudeCodeKernelBenchServer.py (MCP)      GPUs (CUDA devices)
       |                                         |
       +------- SSH tunnel (port forward) -------+
```

Key constraints:
- The MCP server cannot spawn or manage processes on the eval server machine through the tunnel — the tunnel only forwards a port to the eval server process.
- If the eval server process dies, the tunnel stays alive but nothing is listening on the forwarded port.
- Automatic restart logic must run **locally on the eval server machine** (the watchdog).

## Constraints

- **YubiKey physical authentication**: Meta devservers require YubiKey touch for SSH authentication. This makes programmatic SSH impossible — no script or agent can SSH into a devserver without a human physically touching the YubiKey. As a result, remote server management must be done manually via SSH. The watchdog handles all automatic recovery locally on the eval server machine; cases the watchdog can't handle (e.g., watchdog itself crashes, host-level GPU failures) require manual SSH intervention.

## Design

The solution has three components:

- **Server-side** (eval server machine): a watchdog process and a health endpoint handle automatic recovery.
- **Client-side** (dev server): the MCP server retries through brief outages via existing `kbEvalClient.py` retry logic.

### Component 1: `/health` Endpoint (`kbEvalServer.py`)

The server currently has `/stats` and `/info` but no actual health verification. Add a `/health` endpoint that checks:

1. **Process is responsive** — can handle HTTP requests (implicit by responding).
2. **Error rate is acceptable** — `TOTAL_ERROR_COUNTER < MAX_ERROR_COUNT`.
3. **Not approaching time limit** — `elapsed_time < MAX_RUN_TIME`.
4. **Not overloaded** — `parallel_request_counter` below a threshold (optional).

Response format:

```json
// Healthy — 200 OK:
{"status": "healthy", "uptime_seconds": 3600, "error_count": 5, "pending_requests": 2}

// Unhealthy — 503 Service Unavailable:
{"status": "unhealthy", "reason": "error_count_high", "error_count": 48, "pending_requests": 0}
```

HTTP 200 vs 503 lets both the watchdog and remote clients do a simple status-code check.

### Component 2: Local Watchdog (`kbEvalWatchdog.py`)

A Python script that replaces the `while true` shell loop. Runs on the eval server machine.

**Lifecycle:**

```
loop:
  1. Pre-start GPU health check (nvidia-smi)
  2. Start kbEvalServer.py as a subprocess
  3. Startup probe: poll /health until 200 (timeout after 60s; kill and retry if it never comes up)
  4. Liveness loop (while process is alive):
     a. Poll /health every 5s
     b. If unhealthy for 3 consecutive checks (15s):
        - Log the failure reason
        - Kill the server (SIGTERM, wait 10s, SIGKILL)
        - Break to restart
  5. Process exited or was killed
  6. GPU cleanup: check nvidia-smi, kill orphan CUDA processes if needed
  7. Sleep 5s, go to step 1
```

Key behaviors:
- **Startup probe** catches servers that start but never become ready.
- **Liveness probe** catches servers that are alive but stuck — something the shell loop cannot detect.
- **Graceful shutdown** sends SIGTERM before SIGKILL so the server can flush wandb logs.
- **GPU cleanup** prevents cascading failures from leaked GPU memory.
- **Forwards all CLI args** — drop-in replacement for the existing invocation.

Usage:

```bash
# Before:
while true; do python kbEvalServer.py --local_host --port 5676 --device 7; sleep 5; done

# After:
python kbEvalWatchdog.py --local_host --port 5676 --device 7
```

### Component 3: MCP Server Retry Improvements (`claudeCodeKernelBenchServer.py`)

The MCP server should absorb brief eval server outages (during watchdog restarts) so the Claude agent never sees them.

`kbEvalClient.py` already has retry logic (4 retries, exponential backoff: 3s, 9s, 27s, 81s), and it catches all `Exception` types in its retry path, which covers `ConnectionRefusedError` / `httpx.ConnectError` (server process down).

With the watchdog restarting the server in ~10-15s, the existing 4-retry backoff covers most outages without changes on the MCP side.

## Implementation Order

1. **`/health` endpoint** — small addition to `kbEvalServer.py`. Prerequisite for the watchdog.
2. **`kbEvalWatchdog.py`** — replaces the shell loop. This is the core reliability fix.
3. **Verify client retry behavior** — confirm `kbEvalClient.py` retries on connection-level failures (verified: catches all `Exception` in the retry loop).
4. **Update `Makefile`** — change `make kbEval` target to use the watchdog.

## Open Questions

- Should the watchdog monitor GPU memory usage (via `nvidia-smi --query-gpu=memory.used`) and proactively restart if memory is leaking?
- Should the watchdog support managing multiple server instances (different ports, different GPU subsets)?
- What's the right failure threshold? 3 consecutive failed health checks (15s) seems reasonable, but may need tuning for long-running evals that temporarily block the event loop.
