# /// script
# dependencies = ["inspect-ai", "matplotlib", "numpy", "pandas", "pyarrow", "resvg-py", "scipy", "uharfbuzz"]
# ///
"""Generate the editable HTML, curve assets, and Overleaf PNG headline figure."""

from __future__ import annotations

import json
import shutil
from html import escape
from pathlib import Path

import uharfbuzz as hb
import numpy as np
import pandas as pd
from resvg_py import svg_to_bytes
from ridgeline import collect_plot_data, save_distribution_asset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(__file__).with_suffix(".json")
ANALYSIS = REPO_ROOT / "paper/artifacts/transfer_tables/analysis.parquet"
HTML_PATH = Path(__file__).with_suffix(".html")
OUTPUT = REPO_ROOT / "images/example.png"
CURVE_DIR = REPO_ROOT / "images"
ARTIFACTS = REPO_ROOT / "paper/artifacts/figures"
ARCHIVE = ARTIFACTS / "headline_figure.png"
WIDTH, HEIGHT, SCALE = 1440, 725, 2

REGULAR = Path(r"C:\Windows\Fonts\times.ttf")
BOLD = Path(r"C:\Windows\Fonts\timesbd.ttf")
MONO_BOLD = Path(r"C:\Windows\Fonts\consolab.ttf")
DEVANAGARI = Path(r"C:\Windows\Fonts\Nirmala.ttf")
DEVANAGARI_BOLD = Path(r"C:\Windows\Fonts\NirmalaB.ttf")
FONT_FILES = [REGULAR, BOLD, MONO_BOLD, DEVANAGARI, DEVANAGARI_BOLD]

CHIP_CLASSES = {
    "name1": ("name-1-box", "name-1-text"),
    "name2": ("name-2-box", "name-2-text"),
    "count": ("count-box", "count-text"),
    "spot": ("spot-box", "spot-text"),
}


class Font:
    def __init__(self, path: Path, size: float):
        data = path.read_bytes()
        self.font = hb.Font(hb.Face(data))
        self.font.scale = (round(size * 64), round(size * 64))
        hb.ot_font_set_funcs(self.font)

    def measure(self, text: str) -> float:
        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        hb.shape(self.font, buffer)
        return sum(position.x_advance for position in buffer.glyph_positions) / 64


def text(x: float, y: float, value: str, css: str, *, anchor: str | None = None) -> str:
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text class="{css}" x="{x:.1f}" y="{y:.1f}"{anchor_attr}>{escape(value)}</text>'


def rich_text(
    spans: list[tuple[str, str | None]],
    *,
    x: float,
    top: float,
    max_width: float,
    line_height: float,
    regular: Font,
    chip_font: Font,
    text_class: str,
    chip_text_class: str,
) -> tuple[str, float]:
    output: list[str] = []
    x0 = cursor = x
    line_top = top
    space = regular.measure(" ")
    baseline_offset = line_height * 0.72
    chip_height = line_height - 3

    def newline() -> None:
        nonlocal cursor, line_top
        cursor = x0
        line_top += line_height

    for value, style in spans:
        if style == "nowrap":
            width = regular.measure(value)
            if cursor > x0 and cursor + width > x0 + max_width:
                newline()
            output.append(text(cursor, line_top + baseline_offset, value, text_class))
            cursor += width + space
            continue

        if style:
            width = chip_font.measure(value) + 10
            if cursor > x0 and cursor + width > x0 + max_width:
                newline()
            box_class, color_class = CHIP_CLASSES[style]
            output.append(
                f'<rect class="chip-box {box_class}" x="{cursor:.1f}" y="{line_top:.1f}" '
                f'width="{width:.1f}" height="{chip_height:.1f}"/>'
            )
            output.append(
                text(
                    cursor + width / 2,
                    line_top + chip_height / 2,
                    value,
                    f"{chip_text_class} {color_class}",
                    anchor="middle",
                )
            )
            cursor += width + 2
            continue

        for word in value.split():
            width = regular.measure(word)
            if cursor > x0 and cursor + width > x0 + max_width:
                newline()
            output.append(text(cursor, line_top + baseline_offset, word, text_class))
            cursor += width + space

    return "\n".join(output), line_top + line_height


def answer(values: dict[str, object], *, devanagari: bool = False) -> str:
    formula = (
        f"({values['k1']} + {values['k2']}) / "
        f"({values['n1']} + {values['n2']}) × 100 = "
        f"{round((int(values['k1']) + int(values['k2'])) / (int(values['n1']) + int(values['n2'])) * 100)}%"
    )
    if not devanagari:
        return formula
    return formula.translate(str.maketrans("0123456789", "०१२३४५६७८९"))


def card(
    *,
    x: float,
    stage: str,
    spans: list[tuple[str, str | None]],
    answer_label: str,
    answer_text: str,
    max_width: float,
    line_height: float,
    regular: Font,
    chip_font: Font,
    text_class: str,
    chip_text_class: str,
    answer_class: str = "latin answer",
) -> str:
    body, body_bottom = rich_text(
        spans,
        x=x + 26,
        top=104,
        max_width=max_width,
        line_height=line_height,
        regular=regular,
        chip_font=chip_font,
        text_class=text_class,
        chip_text_class=chip_text_class,
    )
    divider = body_bottom + 6
    return "\n".join(
        [
            f'<rect class="card" x="{x}" y="25" width="432" height="335" rx="12"/>',
            f'<rect x="{x + 12}" y="25" width="408" height="7" fill="#173b75"/>',
            text(x + 26, 76, stage, "latin stage"),
            body,
            f'<line class="divider" x1="{x + 26}" y1="{divider:.1f}" x2="{x + 406}" y2="{divider:.1f}"/>',
            text(x + 26, divider + 24, answer_label, "latin answer-label"),
            text(x + 26, divider + 54, answer_text, answer_class),
        ]
    )


SVG_STYLE = """
.latin { font-family: "Times New Roman", Times, serif; fill: #152238; }
.devanagari { font-family: "Nirmala UI", "Noto Serif Devanagari", serif; fill: #152238; }
.stage { font-size: 28px; font-weight: 700; }
.body { font-size: 20px; }
.template-body { font-size: 18px; }
.chip-template { font-family: Consolas, "Courier New", monospace; font-size: 16px; font-weight: 700; dominant-baseline: middle; }
.chip-body { font-size: 20px; font-weight: 700; dominant-baseline: middle; }
.answer-label { font-size: 18px; font-weight: 700; fill: #475467; }
.answer { font-size: 20px; font-weight: 700; }
.performance-title { font-size: 29px; font-weight: 700; }
.model-note { font-size: 20px; fill: #475467; }
.accuracy { font-size: 29px; font-weight: 700; text-anchor: middle; }
.drop { font-size: 21px; font-weight: 700; fill: #c61b3c; text-anchor: middle; }
.chart-title { font-size: 25px; font-weight: 700; }
.card { fill: white; stroke: #d7dee8; stroke-width: 1.5; }
.divider { stroke: #d7dee8; stroke-width: 1; }
.name-1-box { fill: #e8f1fb; stroke: #285f9e; }
.name-1-text { fill: #285f9e; }
.name-2-box { fill: #fff0e8; stroke: #a94f18; }
.name-2-text { fill: #a94f18; }
.count-box { fill: #e8f5ec; stroke: #2f7f45; }
.count-text { fill: #2f7f45; }
.spot-box { fill: #f6e9f7; stroke: #82328a; }
.spot-text { fill: #82328a; }
.chip-box { stroke-width: 1; rx: 5; }
"""


def load_series(performance: dict[str, object]) -> list[dict[str, object]]:
    model = str(performance["model"])
    n_sets = int(performance["resampled_sets"])
    seed = int(performance.get("seed", 42))
    problems = pd.read_parquet(ANALYSIS, columns=["model", "language", "split", "source_id", "correct"])
    problems = problems[problems["model"] == model]
    curves, stats = collect_plot_data(problems, ["eng", "mar"], n_sets, seed)
    stats_by_language = {row.language: row for row in stats}

    original = problems[(problems["language"] == "eng") & (problems["split"] == "original")]["correct"].to_numpy(float)
    original_sets = np.random.default_rng(seed).choice(original, size=(n_sets, len(original)), replace=True).mean(axis=1)
    return [
        {"key": "original", "title": "Original English benchmark", "values": original_sets, "mean": float(original_sets.mean())},
        {"key": "english", "title": "Sampled English variants", "values": curves["eng"][0], "mean": stats_by_language["eng"].synthetic_mean},
        {"key": "marathi", "title": "Matched Marathi variants", "values": curves["mar"][0], "mean": stats_by_language["mar"].synthetic_mean},
    ]


def build_html(data: dict[str, object], series: list[dict[str, object]]) -> tuple[str, str]:
    template = data["template"]
    sample = data["sample"]
    performance = data["performance"]

    template_spans = [
        (f"{{name1={template['name1']}}}", "name1"),
        ("'s dog has", "nowrap"),
        (f"{{n1={template['n1']}}}", "count"),
        (" puppies, ", None),
        (f"{{k1={template['k1']}}}", "spot"),
        (" of which have spots. ", None),
        (f"{{name2={template['name2']}}}", "name2"),
        ("'s dog has", "nowrap"),
        (f"{{n2={template['n2']}}}", "count"),
        (" puppies, ", None),
        (f"{{k2={template['k2']}}}", "spot"),
        (" of which have spots. What percentage of all the puppies have spots?", None),
    ]
    sample_spans = [
        (str(sample["name1"]), "name1"),
        ("'s dog has", "nowrap"),
        (str(sample["n1"]), "count"),
        (" puppies, ", None),
        (str(sample["k1"]), "spot"),
        (" of which have spots. ", None),
        (str(sample["name2"]), "name2"),
        ("'s dog has", "nowrap"),
        (str(sample["n2"]), "count"),
        (" puppies, ", None),
        (str(sample["k2"]), "spot"),
        (" of which have spots. What percentage of all the puppies have spots?", None),
    ]
    marathi_spans = [
        (str(sample["translated_name1"]), "name1"),
        ("च्या कुत्रीला", "nowrap"),
        (str(sample["n1"]).translate(str.maketrans("0123456789", "०१२३४५६७८९")), "count"),
        (" पिल्ले आहेत, त्यापैकी ", None),
        (str(sample["k1"]).translate(str.maketrans("0123456789", "०१२३४५६७८९")), "spot"),
        (" पिल्ल्यांवर ठिपके आहेत. ", None),
        (str(sample["translated_name2"]), "name2"),
        ("च्या कुत्रीला", "nowrap"),
        (str(sample["n2"]).translate(str.maketrans("0123456789", "०१२३४५६७८९")), "count"),
        (" पिल्ले आहेत, त्यापैकी ", None),
        (str(sample["k2"]).translate(str.maketrans("0123456789", "०१२३४५६७८९")), "spot"),
        (" पिल्ल्यांवर ठिपके आहेत. सर्व पिल्ल्यांपैकी किती टक्के पिल्ल्यांवर ठिपके आहेत?", None),
    ]

    cards = "\n".join(
        [
            card(
                x=38,
                stage="1 · ORIGINAL → TEMPLATE",
                spans=template_spans,
                answer_label="ORIGINAL ANSWER",
                answer_text=answer(template),
                max_width=380,
                line_height=30,
                regular=Font(REGULAR, 18),
                chip_font=Font(MONO_BOLD, 16),
                text_class="latin template-body",
                chip_text_class="chip-template",
            ),
            card(
                x=504,
                stage="2 · SAMPLE NEW VALUES",
                spans=sample_spans,
                answer_label="RENDERED ANSWER",
                answer_text=answer(sample),
                max_width=350,
                line_height=34,
                regular=Font(REGULAR, 20),
                chip_font=Font(BOLD, 20),
                text_class="latin body",
                chip_text_class="latin chip-body",
            ),
            card(
                x=970,
                stage="3 · TRANSLATE",
                spans=marathi_spans,
                answer_label="SAME RENDERED ANSWER",
                answer_text=answer(sample, devanagari=True),
                max_width=380,
                line_height=34,
                regular=Font(DEVANAGARI, 20),
                chip_font=Font(DEVANAGARI_BOLD, 20),
                text_class="devanagari body",
                chip_text_class="devanagari chip-body",
                answer_class="devanagari answer",
            ),
        ]
    )

    means = [float(item["mean"]) for item in series]
    accuracies = "\n".join(
        text(center, 477, f"{mean:.1%}", "latin accuracy", anchor="middle")
        for center, mean in zip((254, 720, 1186), means, strict=True)
    )
    drops = "\n".join(
        text(center, 475, f"−{(left - right) * 100:.1f} pp →", "latin drop", anchor="middle")
        for center, left, right in zip((487, 953), means[:-1], means[1:], strict=True)
    )
    charts = "\n".join(
        text(x, 536, str(item["title"]), "latin chart-title")
        + f'\n<image x="{x + 12}" y="555" width="408" height="170" href="../../images/headline_curve_{item["key"]}.png"/>'
        for x, item in zip((38, 504, 970), series, strict=True)
    )
    svg = f'''<svg id="headline-figure" xmlns="http://www.w3.org/2000/svg" width="1440" height="725" viewBox="0 0 1440 725" role="img">
<title>Multilingual GSM-Symbolic generation and performance</title>
<style>{SVG_STYLE}</style>
<rect width="1440" height="725" fill="white"/>
{cards}
{text(487, 188, "→", "latin", anchor="middle").replace('class="latin"', 'class="latin" font-size="30" font-weight="700" fill="#475467"')}
{text(953, 188, "→", "latin", anchor="middle").replace('class="latin"', 'class="latin" font-size="30" font-weight="700" fill="#475467"')}
<line x1="38" y1="380" x2="1402" y2="380" stroke="#152238" stroke-width="2"/>
{text(38, 425, "Exact-answer accuracy across matched 100-problem sets", "latin performance-title")}
{text(1402, 424, f'{performance["model"]} · {int(performance["resampled_sets"]):,} resampled sets per stage', "latin model-note", anchor="end")}
<line class="divider" x1="38" y1="440" x2="1402" y2="440"/>
{accuracies}
{drops}
<line class="divider" x1="38" y1="500" x2="1402" y2="500"/>
{charts}
</svg>'''
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multilingual GSM-Symbolic headline figure</title>
<style>html,body{{width:100%;height:100%;margin:0;background:white}}body{{display:grid;place-items:start center}}#headline-figure{{display:block;width:min(100vw,1440px);height:auto}}</style>
</head><body>{svg}</body></html>'''
    return html, svg


def render() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    series = load_series(data["performance"])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    for item in series:
        save_distribution_asset(np.asarray(item["values"]), CURVE_DIR / f"headline_curve_{item['key']}.png")

    html, svg = build_html(data, series)
    HTML_PATH.write_text(html, encoding="utf-8")
    png = svg_to_bytes(
        svg_string=svg,
        width=WIDTH * SCALE,
        height=HEIGHT * SCALE,
        background="#ffffff",
        font_files=[str(font) for font in FONT_FILES],
        resources_dir=str(HTML_PATH.parent),
        text_rendering="optimize_legibility",
        shape_rendering="geometric_precision",
    )
    OUTPUT.write_bytes(png)
    shutil.copy2(OUTPUT, ARCHIVE)
    print(f"Written: {HTML_PATH}")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    render()
