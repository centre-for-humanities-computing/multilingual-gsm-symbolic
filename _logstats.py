import json
from collections import Counter

entries = [json.loads(l) for l in open("src/multilingual_gsm_symbolic/data/templates/eng/train/generation_log.jsonl", encoding="utf-8")]
recent = [e for e in entries if e["source_id"] >= 132]
kinds = Counter()
samples = {}
for e in recent[-40:]:
    for a in e.get("attempts", []):
        k = a.get("failure_kind")
        if "error" in a:
            kinds[k] += 1
            if k not in samples:
                samples[k] = str(a.get("error", ""))[:150]
print(dict(kinds.most_common()))
for k, s in samples.items():
    print(f"--- {k}: {s}")
