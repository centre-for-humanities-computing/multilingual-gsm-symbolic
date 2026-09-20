"""Offline Inspect scoring, summary, and resume integration check."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip('inspect_ai')

from inspect_ai import Task, eval_set
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import math
from inspect_ai.solver import generate

spec = importlib.util.spec_from_file_location(
    'saturation', Path(__file__).resolve().parents[1]/'src/scripts/glm53_saturation.py'
)
saturation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(saturation)


def test_saturation(tmp_path):
    low, high = saturation.wilson(100, 100)
    assert 0.96 < low < 0.97 and abs(high-1) < 1e-12
    task = Task(name='original_eng', dataset=[Sample(id=1, input='test', target='42',
                metadata={'source_id': 1, 'language': 'eng'})], solver=generate(), scorer=math())
    with pytest.raises(ValueError, match='expected 100 sources'):
        saturation.validate_task(task, 'original')
    model = get_model('mockllm/model', custom_outputs=[
        ModelOutput.from_content('mockllm', r'work \boxed{42.0}')
    ])
    success, logs = eval_set(task, model=model, log_dir=str(tmp_path/'eval'), display='none')
    assert success and read_eval_log(logs[0].location).samples[0].scores['math'].value == 'C'
    saturation.report(logs, tmp_path)
    assert 'False' in (tmp_path/'summary.csv').read_text()  # No accuracy for an incomplete split.
    assert (tmp_path/'errors.json').read_text() == '[]'
    success, resumed = eval_set(task, model=model, log_dir=str(tmp_path/'eval'), display='none')
    assert success and resumed[0].eval.run_id == logs[0].eval.run_id
