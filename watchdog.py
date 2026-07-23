import time
import subprocess

COMMAND = "tmux new-session -d -s translate_session \"bash -l -c 'cd /work/translatetemplates/multilingual-gsm-symbolic && uv run src/scripts/translate_templates.py --to all --subfolder symbolic 2>&1 | tee -a translate_symbolic.log'\""

print("Watchdog started. Monitoring translate_templates.py...")

while True:
    try:
        # Check if the process is running
        result = subprocess.run(["pgrep", "-f", "src/scripts/translate_templates.py"], capture_output=True, text=True)
        if not result.stdout.strip():
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Translation process died. Restarting...")
            subprocess.run(COMMAND, shell=True)
    except Exception as e:
        print(f"Error: {e}")
    time.sleep(300)  # Check every 5 minutes
