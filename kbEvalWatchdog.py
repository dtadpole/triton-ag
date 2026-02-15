"""
kbEvalWatchdog.py — Drop-in replacement for the `while true; do python kbEvalServer.py ...; sleep 5; done` pattern.

Manages the kbEvalServer lifecycle with:
- Pre-start GPU health checks (nvidia-smi)
- Startup probes (poll /health until ready)
- Liveness probes (continuous /health polling)
- Graceful shutdown (SIGTERM → wait → SIGKILL)
- Signal forwarding (SIGINT/SIGTERM → child)

Usage:
    python kbEvalWatchdog.py --local_host --port 5676 --device 0
    python kbEvalWatchdog.py --local_host --port 5676 --device 0,1,2,3

All arguments not consumed by the watchdog are forwarded to kbEvalServer.py unchanged.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error


def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [watchdog] {msg}", flush=True)


def check_gpu_health() -> bool:
    """Run nvidia-smi to verify GPUs are accessible."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log(f"nvidia-smi failed (exit {result.returncode}): {result.stderr.strip()}")
            return False
        gpu_lines = result.stdout.strip().split("\n")
        log(f"GPU check passed — {len(gpu_lines)} GPU(s) visible")
        for line in gpu_lines:
            log(f"  {line.strip()}")
        return True
    except FileNotFoundError:
        log("nvidia-smi not found — skipping GPU check")
        return True  # Allow running without nvidia-smi (e.g., CPU-only dev)
    except subprocess.TimeoutExpired:
        log("nvidia-smi timed out (30s) — GPU may be hung")
        return False


def poll_health(port: int, timeout: float = 5.0) -> dict | None:
    """Poll the /health endpoint. Returns parsed JSON on success, None on failure."""
    url = f"http://localhost:{port}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json
            data = json.loads(resp.read().decode())
            return {"status_code": resp.status, **data}
    except Exception:
        return None


def kill_process(proc: subprocess.Popen, grace_period: float = 10.0):
    """Send SIGTERM, wait for grace period, then SIGKILL if still alive."""
    if proc.poll() is not None:
        return  # Already dead

    log(f"Sending SIGTERM to server (pid {proc.pid})")
    try:
        proc.terminate()
    except OSError:
        return

    try:
        proc.wait(timeout=grace_period)
        log(f"Server exited after SIGTERM (exit code {proc.returncode})")
    except subprocess.TimeoutExpired:
        log(f"Server did not exit after {grace_period}s — sending SIGKILL")
        try:
            proc.kill()
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def startup_probe(proc: subprocess.Popen, port: int, timeout: float, poll_interval: float = 2.0) -> bool:
    """Poll /health until the server is ready. Returns True if healthy, False if timed out or process died."""
    log(f"Startup probe: waiting up to {timeout}s for server on port {port}")
    deadline = time.time() + timeout

    while time.time() < deadline:
        # Check if process died
        if proc.poll() is not None:
            log(f"Server process died during startup (exit code {proc.returncode})")
            return False

        result = poll_health(port)
        if result is not None:
            status_code = result.get("status_code", 0)
            if status_code == 200:
                log(f"Startup probe passed — server healthy")
                return True
            else:
                log(f"Startup probe: server responded with status {status_code} (not yet healthy)")

        time.sleep(poll_interval)

    log(f"Startup probe timed out after {timeout}s")
    return False


def liveness_loop(
    proc: subprocess.Popen,
    port: int,
    check_interval: float,
    max_consecutive_failures: int,
) -> str:
    """
    Poll /health while the process is alive.
    Returns:
        "exited"    — process exited on its own
        "unhealthy" — too many consecutive health check failures
    """
    consecutive_failures = 0

    while True:
        # Check if process exited
        if proc.poll() is not None:
            log(f"Server process exited (exit code {proc.returncode})")
            return "exited"

        result = poll_health(port)

        if result is None:
            consecutive_failures += 1
            log(f"Health check failed ({consecutive_failures}/{max_consecutive_failures})")
        elif result.get("status_code", 0) != 200:
            consecutive_failures += 1
            reason = result.get("reason", "unknown")
            log(f"Health check unhealthy: {reason} ({consecutive_failures}/{max_consecutive_failures})")
        else:
            if consecutive_failures > 0:
                log(f"Health check recovered after {consecutive_failures} failure(s)")
            consecutive_failures = 0

        if consecutive_failures >= max_consecutive_failures:
            log(f"Server unhealthy for {consecutive_failures} consecutive checks — triggering restart")
            return "unhealthy"

        time.sleep(check_interval)


def infer_port(server_args: list[str]) -> int:
    """Extract --port value from the forwarded server args, defaulting to 8456."""
    for i, arg in enumerate(server_args):
        if arg == "--port" and i + 1 < len(server_args):
            try:
                return int(server_args[i + 1])
            except ValueError:
                pass
    return 8456


def main():
    parser = argparse.ArgumentParser(
        description="Watchdog for kbEvalServer.py",
        # Allow unknown args to be forwarded to kbEvalServer.py
    )
    parser.add_argument("--health_check_interval", type=float, default=5.0,
                        help="Seconds between liveness health checks (default: 5)")
    parser.add_argument("--max_consecutive_failures", type=int, default=3,
                        help="Consecutive health check failures before restart (default: 3)")
    parser.add_argument("--startup_timeout", type=float, default=60.0,
                        help="Seconds to wait for server startup (default: 60)")

    watchdog_args, server_args = parser.parse_known_args()
    port = infer_port(server_args)

    log(f"Starting watchdog — server args: {server_args}")
    log(f"Health check interval: {watchdog_args.health_check_interval}s, "
        f"max consecutive failures: {watchdog_args.max_consecutive_failures}, "
        f"startup timeout: {watchdog_args.startup_timeout}s, "
        f"inferred port: {port}")

    # Track the child process for signal forwarding
    child_proc: subprocess.Popen | None = None

    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        log(f"Received {sig_name} — shutting down")
        if child_proc is not None and child_proc.poll() is None:
            kill_process(child_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    restart_count = 0

    while True:
        restart_count += 1
        log(f"=== Restart cycle {restart_count} ===")

        # Step 1: Pre-start GPU check
        if not check_gpu_health():
            log("GPU health check failed — waiting 30s before retry")
            time.sleep(30)
            continue

        # Step 2: Start kbEvalServer.py as subprocess
        cmd = [sys.executable, "kbEvalServer.py"] + server_args
        log(f"Starting server: {' '.join(cmd)}")

        child_proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            # Inherit environment so CUDA_VISIBLE_DEVICES etc. are passed through
        )
        log(f"Server started (pid {child_proc.pid})")

        # Step 3: Startup probe
        if not startup_probe(child_proc, port, watchdog_args.startup_timeout):
            log("Startup probe failed — killing server")
            kill_process(child_proc)
            time.sleep(5)
            continue

        # Step 4: Liveness loop
        exit_reason = liveness_loop(
            child_proc,
            port,
            watchdog_args.health_check_interval,
            watchdog_args.max_consecutive_failures,
        )

        # Step 5: Handle exit
        if exit_reason == "unhealthy":
            kill_process(child_proc)

        # Step 6: Post-exit GPU check
        check_gpu_health()

        # Step 7: Brief pause before restart
        log("Waiting 5s before restart...")
        time.sleep(5)


if __name__ == "__main__":
    main()
