"""Repair loop for synthetic English templates.

Repeatedly:
1. Deletes train templates marked `ignore = true` (or unparseable).
2. Re-runs createsynthetictemplates.py, which skips already-good templates and
   regenerates the deleted ones (plus the next batch of new questions).
3. Stops when a full pass produces nothing left to generate and no failures.

Goal: 100% of GSM8K train questions templatized with zero ignored templates.
"""

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN_DIR = ROOT / "src" / "multilingual_gsm_symbolic" / "data" / "templates" / "eng" / "train"
DRIVER = ROOT / "createsynthetictemplates.py"
MAX_ROUNDS = int(os.getenv("REPAIR_MAX_ROUNDS", "1000"))
LOCK_FILE = ROOT / "logs" / "repair.lock"
DRIVER_TIMEOUT_SECONDS = float(os.getenv("DRIVER_TIMEOUT_SECONDS", "7200"))


def _pid_alive(pid: int) -> bool:
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if handle:
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    return False


def acquire_lock() -> bool:
    LOCK_FILE.parent.mkdir(exist_ok=True)
    try:
        fd = os.open(str(LOCK_FILE), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(LOCK_FILE.read_text().strip() or "0")
            if _pid_alive(pid):
                print(f"[repair] another repair loop is already running (pid {pid}); exiting.", flush=True)
                return False
        except ValueError:
            pass
        # Stale lock from a dead process: replace it atomically.
        LOCK_FILE.unlink(missing_ok=True)
        return acquire_lock()


def release_lock() -> None:
    LOCK_FILE.unlink(missing_ok=True)


def log(msg: str) -> None:
    from datetime import datetime

    print(f"[repair {datetime.now():%H:%M:%S}] {msg}", flush=True)


def delete_ignored() -> int:
    removed = 0
    for p in sorted(TRAIN_DIR.glob("*.toml")):
        try:
            data = tomllib.loads(p.read_text(encoding="utf-8"))
        except Exception:
            p.unlink()
            removed += 1
            continue
        if data.get("ignore"):
            p.unlink()
            removed += 1
    return removed


def main() -> None:
    if not acquire_lock():
        return
    try:
        _run_loop()
    finally:
        release_lock()


def _run_loop() -> None:
    total_removed = 0
    for round_no in range(1, MAX_ROUNDS + 1):
        removed_before = delete_ignored()
        total_removed += removed_before
        if removed_before:
            log(f"round {round_no}: deleted {removed_before} failed template(s)")

        # Stream driver output live (no capture pipes: a wedged child holding a
        # pipe would stall the loop invisibly). Completion is detected via the
        # driver's status file.
        status_file = ROOT / "logs" / "driver_status.json"
        status_file.write_text("{}", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(DRIVER)],
            cwd=ROOT,
            env=os.environ.copy(),
        )
        try:
            proc.wait(timeout=DRIVER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
            log(f"round {round_no}: driver exceeded {DRIVER_TIMEOUT_SECONDS}s; killed")
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        if status.get("jobs") == 0:
            print("[repair] All English questions templatized.", flush=True)
            break

        removed_after = delete_ignored()
        total_removed += removed_after
        if removed_after:
            log(f"round {round_no}: {removed_after} template(s) failed this round; will retry")

        done_count = sum(1 for _ in TRAIN_DIR.glob("*.toml"))
        log(f"round {round_no}: {done_count} good templates on disk")

        if removed_after == 0 and removed_before == 0 and done_count >= 7400:
            print("[repair] No repairs needed; looks complete.", flush=True)
            break

    log(f"finished; total deleted-and-regenerated over run: {total_removed}")


if __name__ == "__main__":
    main()