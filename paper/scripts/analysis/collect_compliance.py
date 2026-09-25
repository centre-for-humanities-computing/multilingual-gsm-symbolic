"""Per-language answer-format compliance, from the full set of eval logs.

Two distinct measurements:
  has_boxed   did the model emit the \\boxed{} format the prompt asks for
  extracted   could the scorer recover an answer at all -- it also accepts the
              GSM8K "#### n" form, so a missing \\boxed is not automatically a
              lost point. This is the one that costs accuracy.

Three things learned the hard way and encoded here:
  * Over ssh, a backgrounded run dies with the session if the account has
    Linger=no -- setsid/nohup are not enough. Run it in the FOREGROUND of a
    held-open session.
  * Results are appended per log, so a death costs one log, not the run. Rerun
    to resume: logs already in the CSV are skipped.
  * Parsing dominates, not bandwidth (6 MB/s/stream measured). Processes, not
    threads, and a stride over samples -- 500 of 2000 is ample for a rate.

Parsing bypasses inspect_ai's pydantic validation (~100ms/sample) by
decompressing the zstd zip entries directly (~1ms/sample).
"""
import os, re, sys, json, csv, struct, zipfile, zstandard
import concurrent.futures as cf
from huggingface_hub import HfApi, hf_hub_download

RID, REV = "danish-foundation-models/multilingual-gsm-symbolic", "refs/pr/16"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "artifacts", "analysis", "compliance.csv")
STREAM = os.path.join(HERE, "_stream")   # scratch; each log is deleted after parsing
LANGS = {"ara","dan","deu","eng","est","fra","hin","isl","ita","jpn","mar","nld","rus","ukr","zho"}
STRIDE = 4                                  # parse every 4th sample: 500 of 2000
BOXED = re.compile(r"\\boxed\s*\{")
COLS = ["file","model","language","n","has_boxed","extracted","correct",
        "extracted_but_wrong","unparsed"]

def read_entry(z, name):
    zi = z.getinfo(name)
    if zi.compress_type != 93:                 # not zstd: let zipfile handle it
        return z.read(name)
    z.fp.seek(zi.header_offset)
    hdr = z.fp.read(30)
    n, m = struct.unpack("<HH", hdr[26:30])
    z.fp.seek(zi.header_offset + 30 + n + m)
    return zstandard.ZstdDecompressor().decompress(
        z.fp.read(zi.compress_size), max_output_size=zi.file_size)

def one(f):
    lg = re.search(r"symbolic-synthetic-([a-z_]+)@", f).group(1)
    p = None
    try:
        p = hf_hub_download(RID, f, repo_type="dataset", revision=REV, local_dir=STREAM)
        z = zipfile.ZipFile(p); names = z.namelist()
        model = None
        for nm in names:
            if nm.endswith("start.json"):
                j = json.loads(read_entry(z, nm))
                model = (j.get("eval") or {}).get("model") or j.get("model")
                break
        n = box = ext = ok = ext_wrong = bad = 0
        for nm in sorted(x for x in names if x.startswith("samples/"))[::STRIDE]:
            try:
                d = json.loads(read_entry(z, nm))
            except Exception:
                bad += 1; continue
            c = (d.get("output") or {}).get("completion") or ""
            # The scorer is registered as "math", not "pattern"; and a failed
            # extraction is the literal string "None", not an empty field.
            # Reading either wrong silently zeroes every extraction column.
            scores = d.get("scores") or {}
            sc = scores.get("math") or (next(iter(scores.values()), {}) or {})
            a = (sc.get("answer") or "").strip()
            got = bool(a) and a != "None"
            n += 1
            box += bool(BOXED.search(c)); ext += got
            ok += sc.get("value") == "C"
            ext_wrong += got and sc.get("value") != "C"
        z.fp.close()
        return (f, model, lg, n, box, ext, ok, ext_wrong, bad)
    except Exception as e:
        print("ERR", lg, type(e).__name__, str(e)[:70], flush=True)
        return None
    finally:
        if p and os.path.exists(p):
            os.remove(p)

def main():
    api = HfApi()
    want = [f for f in api.list_repo_files(RID, repo_type="dataset", revision=REV)
            if f.endswith(".eval")
            and (m := re.search(r"symbolic-(synthetic)-([a-z_]+)@", f))
            and m.group(2) in LANGS]

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {r["file"] for r in csv.DictReader(fh)}
    todo = [f for f in want if f not in done]
    print(f"{len(want)} synthetic logs, {len(done)} already done, {len(todo)} to go",
          flush=True)
    if not todo:
        print("DONE", len(done), "logs", flush=True); return

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new: w.writerow(COLS)
        with cf.ProcessPoolExecutor(max_workers=16) as ex:
            for i, r in enumerate(ex.map(one, todo, chunksize=1), 1):
                if r:
                    w.writerow(r); fh.flush()
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
    print("DONE", len(done) + len(todo), "logs", flush=True)

if __name__ == "__main__":
    main()
