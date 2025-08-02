# arm_watchdog_v2.py
import os
import signal
import time
import multiprocessing as mp
import atexit
from typing import Optional

def _watchdog_worker(target_pgid: int, recv_conn, timeout_sec: float, grace_sec: float):
    """
    Wait for 'A' (arm). If no 'D' (done) before timeout:
      kill target process group: TERM -> wait(grace) -> KILL.
    """
    # Ensure we are NOT in the same group/session as the target.
    try:
        os.setsid()  # new session & pgrp for watchdog
    except Exception:
        pass
    try:
        # Extra safety: don't let TERM kill the watchdog even if setsid failed
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except Exception:
        pass

    # 1) Wait for ARM
    try:
        msg = recv_conn.recv()  # blocks
    except (EOFError, OSError):
        return
    if msg != "A":
        return

    deadline = time.monotonic() + float(timeout_sec)

    # 2) Wait for DONE or timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # Escalating kill of the TARGET process group
            try:
                os.killpg(target_pgid, signal.SIGTERM)
            except ProcessLookupError:
                return

            end = time.monotonic() + max(0.0, float(grace_sec))
            while time.monotonic() < end:
                time.sleep(0.05)
                try:
                    os.killpg(target_pgid, 0)  # probe
                except ProcessLookupError:
                    return

            try:
                os.killpg(target_pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return

        if recv_conn.poll(remaining):
            # Any message after ARM is treated as "done/cancel"
            try:
                _ = recv_conn.recv()
            except (EOFError, OSError):
                pass
            return

class ArmableWatchdog:
    """
    Armable watchdog that kills the *process group* of this process on timeout.

    Args:
        timeout_sec: seconds from .arm() to deadline
        grace_sec:   seconds between SIGTERM and SIGKILL
        make_own_pgrp: if True, put this process in its own group so children are killed too.
    """
    def __init__(self, timeout_sec: float, grace_sec: float = 3.0, make_own_pgrp: bool = True):
        self.timeout_sec = float(timeout_sec)
        self.grace_sec = float(grace_sec)
        self._armed = False
        self._send = None  # type: Optional[mp.connection.Connection]
        self._proc = None  # type: Optional[mp.Process]

        if make_own_pgrp:
            try:
                os.setpgrp()  # become process group leader (pgid = pid)
            except Exception:
                pass

        ctx = mp.get_context("spawn")  # avoid copying a wedged interpreter state
        recv, send = ctx.Pipe(duplex=False)
        self._send = send
        target_pgid = os.getpgrp()

        self._proc = ctx.Process(
            target=_watchdog_worker,
            args=(target_pgid, recv, self.timeout_sec, self.grace_sec),
            daemon=True,
            name="ArmWatchdog",
        )
        self._proc.start()
        # Parent owns only the send end
        recv.close()

        atexit.register(self.cancel)

    def __enter__(self):
        return self.arm

    def arm(self):
        if self._send and not self._armed:
            try:
                self._send.send("A")
                self._armed = True
            except (BrokenPipeError, OSError):
                pass

    def cancel(self):
        if self._send:
            try:
                self._send.send("D")
            except (BrokenPipeError, OSError):
                pass
            try:
                self._send.close()
            except Exception:
                pass
            self._send = None

    def __exit__(self, exc_type, exc, tb):
        self.cancel()

# --- Demo ---
if __name__ == "__main__":
    print("Demo: arming for 5 seconds, then sleeping 10s to force a kill")
    with ArmableWatchdog(timeout_sec=5, grace_sec=2) as arm:
        time.sleep(1)
        print("arming now")
        arm()
        time.sleep(10)  # should be killed by watchdog
        print("after sleeping 10s, we are here")
