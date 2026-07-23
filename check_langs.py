import os, glob

DATA_ROOT = "src/multilingual_gsm_symbolic/data/templates"
languages = [
    "afr", "amh", "ara", "asm", "azb", "bam", "bar", "ben", "bho", "bul", "ceb", "ces", "ctg", "dan", "deu", "dyu", "ell", "eng", "est", "fin", "fra", "fuv", "gaz", "gle", "guj", "hat", "hau", "hin", "hrv", "hun", "ibo", "ind", "isl", "ita", "jav", "jpn", "kan", "kaz", "khm", "kmr", "kor", "lav", "lin", "lit", "lug", "mal", "mar", "mai", "mag", "mlt", "mos", "mya", "nld", "npi", "nso", "nya", "ory", "pbu", "pcm", "pes", "pnb", "pol", "por", "ron", "run", "rus", "sck", "sin", "skr", "slk", "slv", "snd", "sna", "som", "spa", "sun", "swh", "swe", "tam", "tel", "tha", "tgl", "tsn", "tur", "ukr", "urd", "uig", "uzn", "vie", "vjk", "wol", "xho", "yor", "yue", "zho", "zlm", "zul", "hne", "kin", "ktu"
]

human_validated = {"ara", "dan", "fra", "hin", "isl", "jpn", "mar", "nld", "rus", "ukr", "zho"}
# English is source ("eng") and "eng_metric" is sometimes source.
# Let's count how many have 100 templates in 'symbolic'.

done = 0
left = 0
for lang in languages:
    if lang == "eng":
        continue
    if lang in human_validated:
        continue # Not bulk translated
    
    tgt_symbolic = os.path.join(DATA_ROOT, lang, "symbolic")
    if os.path.exists(tgt_symbolic):
        tomls = glob.glob(os.path.join(tgt_symbolic, "*.toml"))
        if len(tomls) >= 100:
            done += 1
        else:
            left += 1
            print(f"Pending: {lang} (has {len(tomls)} templates)")
    else:
        left += 1
        print(f"Pending: {lang} (no symbolic folder)")

print(f"Done: {done}, Left: {left}")
