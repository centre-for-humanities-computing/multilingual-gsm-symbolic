import csv
import inspect_ai
import json
import zipfile
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "paper" / "artifacts" / "glm53-hf-pr16"
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
files = sorted((root / "logs").glob("*.eval"))
assert len(files) == 30, len(files)

results = {}
for path in files:
    with zipfile.ZipFile(path) as archive:
        header = json.loads(archive.read("header.json"))
        task = header["eval"]["task"]
        sample_names = [name for name in archive.namelist() if name.startswith("samples/")]
        samples = [json.loads(archive.read(name)) for name in sample_names]
    ids = [sample["id"] for sample in samples]
    scores = [sample.get("scores", {}).get("math") for sample in samples]
    assert len(ids) == len(set(ids)), task
    results[task] = {
        "count": len(samples),
        "scored": sum(score is not None and sample.get("error") is None for sample, score in zip(samples, scores)),
        "correct": sum(score is not None and score.get("value") == "C" for score in scores),
        "status": header["status"],
    }

assert set(results) == set(manifest["tasks"])
for task, expected in manifest["tasks"].items():
    assert results[task]["count"] == expected, (task, results[task])

rows = list(csv.DictReader((root / "summary.csv").open(encoding="utf-8", newline="")))
assert len(rows) == 30
for row in rows:
    task = next(name for name in results if name.endswith(f"/{row['split']}_{row['language']}"))
    item = results[task]
    assert int(row["expected"]) == item["count"]
    assert int(row["scored"]) == item["scored"]
    assert int(row["correct"]) == item["correct"]
    assert int(row["api_errors"]) == item["count"] - item["scored"]

totals = Counter()
for item in results.values():
    totals.update({"count": item["count"], "scored": item["scored"], "correct": item["correct"]})
print(dict(totals))
print("complete_logs", sum(item["status"] == "success" and item["count"] == item["scored"] for item in results.values()))
