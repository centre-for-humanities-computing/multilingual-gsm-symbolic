"""Run the paper's pinned Hugging Face Inspect tasks with GLM-5.3."""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

from inspect_ai import eval_set
from inspect_ai._eval.task.hf import task_create_from_hf
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model

ROOT = Path(__file__).resolve().parents[2]
REPO = 'danish-foundation-models/multilingual-gsm-symbolic'
# HF refs/pr/16, pinned on 2026-09-16; includes localized prompts and math scorer.
REVISION = 'd34a0ffcb2851179ccac891807fa3a29ccd896f6'
LANGUAGES = 'ara dan deu eng est fra hin isl ita jpn mar nld rus ukr zho'.split()
BASE_URL = 'https://ai.cloud.sdu.dk/v1'
MODEL = 'zai-org/GLM-5.3'


def validate_task(task, split):
    counts = Counter(sample.metadata['source_id'] for sample in task.dataset)
    expected_per_source = 1 if split == 'original' else 20
    if len(counts) != 100 or set(counts.values()) != {expected_per_source}:
        raise ValueError(f'{task.name}: expected 100 sources with {expected_per_source} samples each; got {counts}')
    return set(counts)


def load_tasks(languages, revision):
    tasks = []
    for lang in languages:
        source_ids = None
        for split in ('original', 'synthetic'):
            name = f'{split}_{lang}'
            # Use Inspect's own HF loader: dataset, localized solver and scorer from eval.yaml.
            task = task_create_from_hf(f'hf/{REPO}/{name}@{revision}')[0]
            ids = validate_task(task, split)
            if source_ids is not None and ids != source_ids:
                raise ValueError(f'{lang}: original and synthetic source IDs differ')
            source_ids = ids
            tasks.append(task)
            print(f'{name}: {len(task.dataset)} published samples', flush=True)
    return tasks


def wilson(correct, total):
    p, z = correct / total, 1.959963984540054
    denom = 1 + z*z / total
    center = (p + z*z / (2*total)) / denom
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denom
    return center-half, center+half


def report(logs, output):
    summaries, errors = [], []
    for log in logs:
        if log.samples is None:
            log = read_eval_log(log.location)
        task_id = log.eval.task.rsplit('/', 1)[-1]
        split, lang = task_id.split('_', 1)
        expected = 100 if split == 'original' else 2000
        counts, by_template = Counter(), defaultdict(list)
        for sample in log.samples or []:
            score = (sample.scores or {}).get('math')
            if sample.error or score is None:
                counts['api_errors'] += 1
                continue
            counts['scored'] += 1
            counts['correct'] += score.value == 'C'
            counts['parse_failures'] += score.answer in (None, 'None', '')
            counts['truncated'] += sample.output.stop_reason == 'max_tokens'
            by_template[sample.metadata['source_id']].append(int(score.value == 'C'))
            if score.value != 'C':
                errors.append({'language': lang, 'split': split, 'sample_id': sample.id,
                               'source_id': sample.metadata['source_id'], 'target': sample.target,
                               'answer': score.answer, 'stop_reason': sample.output.stop_reason,
                               'response': sample.output.completion})
        complete = log.status == 'success' and counts['scored'] == expected
        lo, hi = None, None
        if complete and split == 'original':
            lo, hi = wilson(counts['correct'], expected)
        elif complete and len(by_template) == 100:
            means = [mean(values) for values in by_template.values()]
            half = 1.984217 * stdev(means) / math.sqrt(100)
            lo, hi = max(0, mean(means)-half), min(1, mean(means)+half)
        summaries.append({'language': lang, 'split': split, 'complete': complete,
                          'expected': expected, 'scored': counts['scored'], 'correct': counts['correct'],
                          'accuracy': counts['correct']/expected if complete else None,
                          'ci_95_low': lo, 'ci_95_high': hi,
                          'interval_method': 'Wilson' if split == 'original' else 'template-cluster t interval',
                          'parse_failures': counts['parse_failures'], 'truncated': counts['truncated'],
                          'api_errors': counts['api_errors']})
    if summaries:
        with (output/'summary.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
    (output/'errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--languages', nargs='+', default=LANGUAGES)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--output', type=Path, default=ROOT/'logs/glm53-hf-pr16')
    parser.add_argument('--dry-run', action='store_true', help='Validate/download HF tasks without model requests')
    parser.add_argument('--smoke', action='store_true', help='One sample per task, separate output folder')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature (default: 1.0)')
    parser.add_argument('--max-tokens', type=int, default=None, help='Maximum output tokens (default: None, no token cap)')
    parser.add_argument('--reasoning-effort', choices=['low', 'high', 'max'], default='max')
    parser.add_argument('--max-connections', type=int, default=64, help='Maximum concurrent HTTP connections (default: 64)')
    parser.add_argument('--max-tasks', type=int, default=4, help='Maximum concurrent tasks (default: 4)')
    parser.add_argument('--max-retries', type=int, default=5, help='Maximum HTTP retries per request (default: 5)')
    parser.add_argument('--retry-attempts', type=int, default=3, help='Task-level retry attempts on error (default: 3)')
    args = parser.parse_args()
    if len(set(args.languages)) != len(args.languages):
        parser.error('Duplicate languages')
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('--revision must be an immutable HF commit SHA')
    tasks = load_tasks(args.languages, args.revision)
    print(f'Validated {sum(len(t.dataset) for t in tasks)} samples in {len(tasks)} Inspect tasks.', flush=True)
    if args.dry_run:
        return
    key = os.environ.get('SDU_API_KEY')
    if not key:
        parser.error('Set SDU_API_KEY or use run_glm53_saturation.ps1')
    output = args.output/'smoke' if args.smoke else args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'model': MODEL, 'base_url': BASE_URL, 'hf_repo': REPO, 'hf_revision': args.revision,
                'languages': args.languages, 'temperature': args.temperature, 'max_tokens': args.max_tokens,
                'reasoning_effort': args.reasoning_effort, 'smoke': args.smoke,
                'tasks': {t.name: len(t.dataset) for t in tasks}}
    manifest_path = output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        parser.error('Run settings changed; use a fresh --output directory')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    model = get_model('openai-api/sdu/'+MODEL, base_url=BASE_URL, api_key=key, responses_api=False)
    success, logs = eval_set(
        tasks, log_dir=str(output/'eval'), model=model, temperature=args.temperature, max_tokens=args.max_tokens,
        extra_body={'reasoning_effort': args.reasoning_effort}, limit=1 if args.smoke else None,
        max_connections=args.max_connections, max_tasks=args.max_tasks, max_retries=args.max_retries,
        timeout=300, retry_attempts=args.retry_attempts, fail_on_error=False, display='plain', log_model_api=False,
    )
    report(logs, output)
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
