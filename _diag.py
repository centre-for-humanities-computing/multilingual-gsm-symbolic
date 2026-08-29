import json

entries = [json.loads(l) for l in open("src/multilingual_gsm_symbolic/data/templates/eng/train/generation_log.jsonl", encoding="utf-8")]
for sid in (135, 243, 252):
    rel = [e for e in entries if e["source_id"] == sid]
    e = rel[-1]
    print("=" * 30, "source", sid, "success:", e.get("success"))
    last = [a for a in e.get("attempts", []) if "error" in a][-1]
    err = last["error"]
    # show the unified diff part
    idx = err.find("---")
    print(err[idx : idx + 1200] if idx != -1 else err[:800])
