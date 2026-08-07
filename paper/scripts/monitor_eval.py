#!/usr/bin/env python3
"""
Monitor script for overnight eval runs.
Checks tmux session, inspects log directory, recovers from crashes, and outputs progress summary.
"""
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = REPO_ROOT / "hf_dataset" / "logs_pr16_d1ac6bc"
TMUX_SESSION = "eval-pr16"
CMD = "uv run paper/scripts/ucloudeval --model all --revision refs/pr/16 --log-dir hf_dataset/logs_pr16_d1ac6bc"

def check_tmux_running() -> bool:
    res = subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION], capture_output=True)
    return res.returncode == 0

def cleanup_vllm():
    subprocess.run(["pkill", "-9", "-f", "vllm serve"], stderr=subprocess.DEVNULL, check=False)

def restart_tmux():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Restarting tmux session '{TMUX_SESSION}'...")
    cleanup_vllm()
    time.sleep(3)
    subprocess.run([
        "tmux", "new-session", "-d", "-s", TMUX_SESSION,
        "-x", "160", "-y", "48", CMD
    ], cwd=str(REPO_ROOT), check=True)

def get_completed_eval_files() -> list[Path]:
    if not LOG_DIR.exists():
        return []
    return list(LOG_DIR.glob("*.eval"))

def main():
    print(f"=== Eval Monitoring Report ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===")
    tmux_ok = check_tmux_running()
    print(f"Tmux session '{TMUX_SESSION}' active: {tmux_ok}")
    
    eval_files = get_completed_eval_files()
    print(f"Completed eval result files in {LOG_DIR.name}: {len(eval_files)}")
    
    if not tmux_ok:
        print("WARNING: Tmux session stopped! Attempting restart...")
        restart_tmux()
        time.sleep(2)
        print(f"Tmux session status post-restart: {check_tmux_running()}")
    else:
        # Check last log output
        res = subprocess.run(["tmux", "capture-pane", "-t", TMUX_SESSION, "-p", "-S", "-10"], capture_output=True, text=True)
        if res.returncode == 0:
            lines = res.stdout.strip().splitlines()
            print("Recent tmux log tail:")
            for line in lines[-5:]:
                print(f"  {line}")

if __name__ == "__main__":
    main()
