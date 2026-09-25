"""Run retry of all incomplete tasks concurrently using native eval_set."""

import argparse
import copy
import glob
import os
import shutil
import sys
from pathlib import Path

from inspect_ai import eval_set
from inspect_ai._eval.task.hf import task_create_from_hf
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import get_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm53_saturation import BASE_URL, MODEL, REPO, REVISION, report


def task_name(log):
    return log.eval.task.rsplit('/', 1)[-1].split('@')[0]


def scored_samples(log):
    return {
        sample.id: sample
        for sample in (log.samples or [])
        if sample.error is None and (sample.scores or {}).get('math') is not None
    }


def recover_retry_logs(retry_dir, task_logs):
    """Merge any scored samples left by an interrupted retry run."""
    recovered = {}
    for retry_file in sorted(retry_dir.glob('*.eval')):
        retry_log = read_eval_log(str(retry_file))
        recovered.setdefault(task_name(retry_log), {}).update(scored_samples(retry_log))

    merged = 0
    for name, samples in recovered.items():
        if name not in task_logs:
            continue
        eval_file, log = task_logs[name]
        existing = scored_samples(log)
        new_ids = samples.keys() - existing.keys()
        if not new_ids:
            continue
        log.samples = [samples.get(sample.id, sample) for sample in log.samples or []]
        write_eval_log(log, eval_file)
        merged += len(new_ids)
    return merged

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, default=Path('logs/glm53-hf-pr16/eval'))
    parser.add_argument('--output-dir', type=Path, default=Path('logs/glm53-hf-pr16'))
    parser.add_argument('--max-connections', type=int, default=16)
    parser.add_argument('--max-tasks', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()

    key = os.environ.get('SDU_API_KEY')
    if not key:
        print("ERROR: SDU_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    model = get_model('openai-api/sdu/' + MODEL, base_url=BASE_URL, api_key=key, responses_api=False)

    eval_files = sorted(
        path for path in glob.glob(str(args.eval_dir / '*.eval'))
        if not Path(path).name.startswith('retry__')
    )
    if not eval_files:
        print(f"No .eval files found in {args.eval_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {len(eval_files)} evaluation log files...", flush=True)

    task_logs = {}
    missing_by_task = {}
    retry_tasks = []
    task_expected = {}

    for ef in eval_files:
        log = read_eval_log(ef)
        task_id = task_name(log)
        split, lang = task_id.split('_', 1)
        name = f"{split}_{lang}"
        expected = 100 if split == 'original' else 2000
        task_logs[name] = (ef, log)
        task_expected[name] = expected

    retry_tmp_dir = args.output_dir / 'retry_tmp'
    backup_dir = Path(r"C:\g53-handoff\completed-samples-backup")
    for rdir in [retry_tmp_dir, backup_dir]:
        if rdir.exists():
            recovered = recover_retry_logs(rdir, task_logs)
            print(f"Recovered {recovered} scored samples from {rdir}.", flush=True)

    for task_name_value, (ef, log) in task_logs.items():
        split, lang = task_name_value.split('_', 1)
        expected = task_expected[task_name_value]
        failed_sample_ids = []
        for sample in log.samples or []:
            score = (sample.scores or {}).get('math')
            if sample.error is not None or score is None:
                failed_sample_ids.append(sample.id)

        if failed_sample_ids:
            missing_by_task[task_name_value] = failed_sample_ids
            hf_task = task_create_from_hf(f'hf/{REPO}/{task_name_value}@{REVISION}')[0]
            missing_samples = []
            for s_id in failed_sample_ids:
                idx = s_id - 1
                sample = copy.deepcopy(hf_task.dataset[idx])
                sample.id = s_id
                missing_samples.append(sample)
            
            retry_task = copy.copy(hf_task)
            retry_task.dataset = MemoryDataset(samples=missing_samples, name=hf_task.dataset.name)
            retry_tasks.append((task_name_value, retry_task))

    total_missing = sum(len(ids) for ids in missing_by_task.values())
    print(f"\nFound {total_missing} missing samples across {len(retry_tasks)} incomplete tasks.", flush=True)

    if not retry_tasks:
        print("All tasks are already complete!", flush=True)
        for ef, log in task_logs.values():
            log.status = 'success'
            write_eval_log(log, ef)
        report([log for ef, log in task_logs.values()], args.output_dir)
        return

    print(f"Running all {len(retry_tasks)} retry tasks concurrently with {args.max_connections} connections and {args.max_tasks} tasks...\n", flush=True)

    if retry_tmp_dir.exists():
        shutil.rmtree(retry_tmp_dir)
    retry_tmp_dir.mkdir(parents=True, exist_ok=True)

    tasks_to_run = [t[1] for t in retry_tasks]
    # The outer runner retries only unscored IDs. Prevent eval_set from replaying
    # an entire task when one sample has a server error.
    success, retry_eval_logs = eval_set(
        tasks_to_run, log_dir=str(retry_tmp_dir), model=model, temperature=1.0, max_tokens=None,
        extra_body={'reasoning_effort': 'max'}, max_connections=args.max_connections,
        max_tasks=args.max_tasks, max_retries=10, retry_attempts=0, fail_on_error=False,
        timeout=args.timeout, display='plain',
        log_model_api=False, log_samples=True
    )

    # First back up any eval files in retry_tmp so no scored samples are lost
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rf in retry_tmp_dir.glob("*.eval"):
        shutil.copy2(rf, backup_dir / rf.name)

    print(f"\nMerging results into original .eval logs...", flush=True)
    for rdir in [retry_tmp_dir, backup_dir]:
        if rdir.exists():
            recovered = recover_retry_logs(rdir, task_logs)
            print(f"Merged {recovered} scored samples from {rdir}.", flush=True)

    all_final_logs = []
    for name, (ef, log) in task_logs.items():
        scored_count = sum(1 for s in log.samples if not s.error and (s.scores or {}).get('math') is not None)
        log.status = 'success' if scored_count == task_expected[name] else 'error'
        write_eval_log(log, ef)
        print(f"  [{name}] Status: {scored_count}/{task_expected[name]} scored.", flush=True)
        all_final_logs.append(log)

    print(f"\nGenerating final summary report and error log...", flush=True)
    report(all_final_logs, args.output_dir)
    print(f"Finished! Written to {args.output_dir}/summary.csv and {args.output_dir}/errors.json")

    # Clean up retry_tmp
    if retry_tmp_dir.exists():
        shutil.rmtree(retry_tmp_dir)

if __name__ == '__main__':
    main()
