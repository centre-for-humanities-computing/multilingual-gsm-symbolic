# /// script
# dependencies = [
#   "freetype-py",
#   "inspect-ai",
#   "numpy",
#   "pandas",
#   "pillow",
#   "pyarrow",
#   "scipy",
#   "uharfbuzz",
# ]
# ///
"""Render the paper's headline figure without a browser."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import freetype
import numpy as np
import pandas as pd
import uharfbuzz as hb
from eval_log_utils import normal_curve, sample_synthetic_sets
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = REPO_ROOT / "paper/artifacts/transfer_tables/analysis.parquet"
OUTPUT = REPO_ROOT / "paper/artifacts/figures/headline_figure.png"
MODEL = "Qwen2.5-7B-Instruct"
N_SETS = 2_000
SEED = 42
SCALE = 2
WIDTH, HEIGHT = 1440, 725

FONT_DIR = Path("C:/Windows/Fonts")
REGULAR = FONT_DIR / "times.ttf"
BOLD = FONT_DIR / "timesbd.ttf"
MONO_BOLD = FONT_DIR / "consolab.ttf"
DEVANAGARI = FONT_DIR / "Nirmala.ttf"
DEVANAGARI_BOLD = FONT_DIR / "NirmalaB.ttf"

INK = "#152238"
MUTED = "#475467"
LINE = "#D7DEE8"
BLUE = "#173B75"
BLUE_FILL = "#AFC0DD"
BLUE_WASH = "#DCE5F2"
RED = "#C61B3C"
GRID = "#E7EBF1"
AXIS = "#8B97A8"

CHIPS = {
    "name1": ("#285F9E", "#E8F1FB"),
    "name2": ("#A94F18", "#FFF0E8"),
    "count": ("#2F7F45", "#E8F5EC"),
    "spot": ("#82328A", "#F6E9F7"),
}


def p(value: float) -> int:
    return round(value * SCALE)


def pil_font(path: Path, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), p(size))


class ShapedFont:
    """Small HarfBuzz/FreeType adapter for correctly shaped Devanagari."""

    def __init__(self, path: Path, size: float):
        self.data = path.read_bytes()
        self.hb_font = hb.Font(hb.Face(self.data))
        self.hb_font.scale = (p(size) * 64, p(size) * 64)
        hb.ot_font_set_funcs(self.hb_font)
        self.face = freetype.Face(str(path))
        self.face.set_pixel_sizes(0, p(size))
        self.ascender = self.face.size.ascender / 64
        self.height = self.face.size.height / 64

    def shape(self, text: str) -> tuple[list[Any], list[Any]]:
        buffer = hb.Buffer()
        buffer.add_str(text)
        buffer.guess_segment_properties()
        hb.shape(self.hb_font, buffer)
        return buffer.glyph_infos, buffer.glyph_positions

    def measure(self, text: str) -> float:
        _infos, positions = self.shape(text)
        return sum(position.x_advance for position in positions) / 64

    def draw(self, image: Image.Image, xy: tuple[float, float], text: str, fill: str) -> None:
        infos, positions = self.shape(text)
        pen_x = float(xy[0])
        baseline = float(xy[1]) + self.ascender
        color = ImageColor.getrgb(fill)
        for info, position in zip(infos, positions, strict=True):
            self.face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_TARGET_NORMAL)
            slot = self.face.glyph
            bitmap = slot.bitmap
            if bitmap.width and bitmap.rows:
                mask = Image.frombytes("L", (bitmap.width, bitmap.rows), bytes(bitmap.buffer))
                glyph = Image.new("RGBA", mask.size, (*color, 255))
                glyph.putalpha(mask)
                x = round(pen_x + position.x_offset / 64 + slot.bitmap_left)
                y = round(baseline - position.y_offset / 64 - slot.bitmap_top)
                image.alpha_composite(glyph, (x, y))
            pen_x += position.x_advance / 64


class ImageColor:
    @staticmethod
    def getrgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def measure(font: ImageFont.FreeTypeFont | ShapedFont, text: str) -> float:
    return font.measure(text) if isinstance(font, ShapedFont) else font.getlength(text)


def draw_text(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ShapedFont,
    fill: str = INK,
    anchor: str = "lt",
) -> None:
    if isinstance(font, ShapedFont):
        if anchor != "lt":
            raise ValueError("Shaped text currently supports only left-top anchoring.")
        font.draw(image, xy, text, fill)
    else:
        draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def draw_rich_text(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    spans: list[tuple[str, str | None]],
    xy: tuple[int, int],
    max_width: int,
    regular: ImageFont.FreeTypeFont | ShapedFont,
    bold: ImageFont.FreeTypeFont | ShapedFont,
    line_height: int,
) -> int:
    x0, x = xy[0], xy[0]
    y = xy[1]
    for text, style in spans:
        pieces = [text] if style else re.findall(r"\S+\s*|\s+", text)
        for piece in pieces:
            font = bold if style else regular
            padding_x = p(5) if style else 0
            width = measure(font, piece) + 2 * padding_x
            if x > x0 and x + width > x0 + max_width:
                x = x0
                y += line_height
                piece = piece.lstrip()
                width = measure(font, piece) + 2 * padding_x
            if not piece:
                continue
            if style:
                color, background = CHIPS[style]
                draw.rounded_rectangle(
                    (x, y - p(1), x + width, y + line_height - p(4)),
                    radius=p(4),
                    fill=background,
                    outline=color,
                    width=p(1),
                )
                draw_text(image, draw, (x + padding_x, y), piece, font, color)
            else:
                draw_text(image, draw, (x, y), piece, font, INK)
            x += width
    return y + line_height


def load_distributions() -> dict[str, np.ndarray]:
    columns = ["source_id", "language", "correct", "model", "split"]
    rows = pd.read_parquet(ANALYSIS, columns=columns)
    rows = rows[rows["model"] == MODEL]
    rng = np.random.default_rng(SEED)
    original = rows[(rows["language"] == "eng") & (rows["split"] == "original")]["correct"].to_numpy(dtype=float)
    if len(original) != 100:
        raise ValueError(f"Expected 100 original English problems, found {len(original)}")
    distributions = {"original": rng.choice(original, size=(N_SETS, len(original)), replace=True).mean(axis=1)}
    for key, language in (("english", "eng"), ("marathi", "mar")):
        synthetic = rows[(rows["language"] == language) & (rows["split"] == "synthetic")]
        if synthetic["source_id"].nunique() != 100:
            raise ValueError(f"Expected 100 {language} templates, found {synthetic['source_id'].nunique()}")
        distributions[key], _ = sample_synthetic_sets(synthetic, N_SETS, rng)
    return distributions


def draw_card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=p(12), fill="white", outline=LINE, width=p(1.5))
    draw.line((box[0] + p(12), box[1] + p(4), box[2] - p(12), box[1] + p(4)), fill=BLUE, width=p(7))


def draw_plot(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: np.ndarray,
    tick_font: ImageFont.FreeTypeFont,
) -> None:
    left, top, right, bottom = box
    baseline = bottom - p(30)
    curve_x, curve_density, mean, _std = normal_curve(values)
    hist_density, hist_edges = np.histogram(values, bins=18, density=True)
    peak = max(float(curve_density.max()), float(hist_density.max())) * 1.08

    def xscale(value: float) -> int:
        return round(left + value * (right - left))

    def yscale(value: float) -> int:
        return round(baseline - value / peak * (baseline - top))

    for tick in np.linspace(0, 1, 5):
        x = xscale(float(tick))
        draw.line((x, top, x, baseline), fill=GRID, width=p(1))

    points = [(xscale(float(x)), yscale(float(y))) for x, y in zip(curve_x, curve_density, strict=True)]
    draw.polygon([*points, (points[-1][0], baseline), (points[0][0], baseline)], fill=BLUE_WASH)
    for density, x0, x1 in zip(hist_density, hist_edges[:-1], hist_edges[1:], strict=True):
        draw.rectangle(
            (xscale(float(x0)), yscale(float(density)), xscale(float(x1)), baseline),
            fill=BLUE_FILL,
            outline="white",
            width=p(1),
        )
    draw.line(points, fill=BLUE, width=p(2.4), joint="curve")
    mean_x = xscale(mean)
    draw.line((mean_x, top, mean_x, baseline), fill=RED, width=p(2.6))
    mean_y = yscale(float(curve_density.max()))
    draw.ellipse((mean_x - p(6), mean_y - p(6), mean_x + p(6), mean_y + p(6)), fill=BLUE)
    draw.line((left, baseline, right, baseline), fill=AXIS, width=p(1.2))

    for index, tick in enumerate(np.linspace(0, 1, 5)):
        x = xscale(float(tick))
        draw.line((x, baseline, x, baseline + p(6)), fill=AXIS, width=p(1.2))
        anchor = "la" if index == 0 else "ra" if index == 4 else "ma"
        draw.text((x, baseline + p(9)), f"{tick:.0%}", font=tick_font, fill=MUTED, anchor=anchor)


def render() -> None:
    distributions = load_distributions()
    image = Image.new("RGBA", (p(WIDTH), p(HEIGHT)), "white")
    draw = ImageDraw.Draw(image)

    stage_font = pil_font(BOLD, 28)
    body = pil_font(REGULAR, 20)
    body_bold = pil_font(BOLD, 20)
    small_bold = pil_font(BOLD, 18)
    template_body = pil_font(REGULAR, 18)
    template_bold = pil_font(MONO_BOLD, 16)
    answer = pil_font(BOLD, 20)
    performance_title = pil_font(BOLD, 29)
    model_font = pil_font(REGULAR, 20)
    sequence_value = pil_font(BOLD, 29)
    drop_font = pil_font(BOLD, 21)
    chart_title = pil_font(BOLD, 25)
    tick_font = pil_font(REGULAR, 22)
    marathi = ShapedFont(DEVANAGARI, 20)
    marathi_bold = ShapedFont(DEVANAGARI_BOLD, 20)

    card_y, card_h, card_w, gap = p(25), p(335), p(432), p(34)
    card_x = [p(38) + index * (card_w + gap) for index in range(3)]
    for x in card_x:
        draw_card(draw, (x, card_y, x + card_w, card_y + card_h))
    for x in (p(487), p(953)):
        draw_text(image, draw, (x, p(176)), "→", pil_font(BOLD, 30), MUTED, "mm")

    stages = ["1 · ORIGINAL → TEMPLATE", "2 · SAMPLE NEW VALUES", "3 · TRANSLATE"]
    for x, stage in zip(card_x, stages, strict=True):
        draw_text(image, draw, (x + p(26), p(57)), stage, stage_font)

    original_spans = [
        ("{name1=Jennifer}", "name1"),
        ("'s dog has ", None),
        ("{n1=8}", "count"),
        (" puppies, ", None),
        ("{k1=3}", "spot"),
        (" of which have spots. ", None),
        ("{name2=Brandon}", "name2"),
        ("'s dog has ", None),
        ("{n2=12}", "count"),
        (" puppies, ", None),
        ("{k2=4}", "spot"),
        (" of which have spots. What percentage of all the puppies have spots?", None),
    ]
    variant_spans = [
        ("Olivia", "name1"),
        ("'s dog has ", None),
        ("15", "count"),
        (" puppies, ", None),
        ("5", "spot"),
        (" of which have spots. ", None),
        ("Marcus", "name2"),
        ("'s dog has ", None),
        ("10", "count"),
        (" puppies, ", None),
        ("2", "spot"),
        (" of which have spots. What percentage of all the puppies have spots?", None),
    ]
    marathi_spans = [
        ("सई", "name1"),
        ("च्या कुत्रीला ", None),
        ("१५", "count"),
        (" पिल्ले आहेत, त्यापैकी ", None),
        ("५", "spot"),
        (" पिल्ल्यांवर ठिपके आहेत. ", None),
        ("रोहन", "name2"),
        ("च्या कुत्रीला ", None),
        ("१०", "count"),
        (" पिल्ले आहेत, त्यापैकी ", None),
        ("२", "spot"),
        (" पिल्ल्यांवर ठिपके आहेत. सर्व पिल्ल्यांपैकी किती टक्के पिल्ल्यांवर ठिपके आहेत?", None),
    ]
    body_ends = [
        draw_rich_text(
            image,
            draw,
            original_spans,
            (card_x[0] + p(26), p(105)),
            p(380),
            template_body,
            template_bold,
            p(30),
        ),
        draw_rich_text(image, draw, variant_spans, (card_x[1] + p(26), p(105)), p(350), body, body_bold, p(34)),
        draw_rich_text(image, draw, marathi_spans, (card_x[2] + p(26), p(105)), p(380), marathi, marathi_bold, p(34)),
    ]

    answer_dividers = [end + p(6) for end in body_ends]
    for index, divider_y in enumerate(answer_dividers):
        draw.line(
            (card_x[index] + p(26), divider_y, card_x[index] + card_w - p(26), divider_y),
            fill=LINE,
            width=p(1),
        )
    answer_labels = ["ORIGINAL ANSWER", "RENDERED ANSWER", "SAME RENDERED ANSWER"]
    for index, label in enumerate(answer_labels):
        draw_text(image, draw, (card_x[index] + p(26), answer_dividers[index] + p(9)), label, small_bold, MUTED)
    draw_text(
        image,
        draw,
        (card_x[0] + p(26), answer_dividers[0] + p(37)),
        "(3 + 4) / (8 + 12) × 100 = 35%",
        answer,
    )
    draw_text(
        image,
        draw,
        (card_x[1] + p(26), answer_dividers[1] + p(37)),
        "(5 + 2) / (15 + 10) × 100 = 28%",
        answer,
    )
    marathi_bold.draw(
        image,
        (card_x[2] + p(26), answer_dividers[2] + p(37)),
        "(५ + २) / (१५ + १०) × १०० = २८%",
        INK,
    )

    draw.line((p(38), p(380), p(1402), p(380)), fill=INK, width=p(2))
    draw_text(image, draw, (p(38), p(404)), "Exact-answer accuracy across matched 100-problem sets", performance_title)
    draw_text(
        image,
        draw,
        (p(1402), p(411)),
        "Qwen2.5-7B-Instruct · 2,000 resampled sets per stage",
        model_font,
        MUTED,
        "rt",
    )
    draw.line((p(38), p(440), p(1402), p(440)), fill=LINE, width=p(1))
    draw.line((p(38), p(500), p(1402), p(500)), fill=LINE, width=p(1))

    means = {key: float(values.mean()) for key, values in distributions.items()}
    centers = [x + card_w // 2 for x in card_x]
    for center, key in zip(centers, ("original", "english", "marathi"), strict=True):
        draw_text(image, draw, (center, p(455)), f"{means[key]:.1%}", sequence_value, anchor="mt")
    draw_text(
        image,
        draw,
        (p(487), p(457)),
        f"−{(means['original'] - means['english']) * 100:.1f} pp →",
        drop_font,
        RED,
        "mt",
    )
    draw_text(
        image,
        draw,
        (p(953), p(457)),
        f"−{(means['english'] - means['marathi']) * 100:.1f} pp →",
        drop_font,
        RED,
        "mt",
    )

    chart_titles = ["Original English benchmark", "Sampled English variants", "Matched Marathi variants"]
    for x, label in zip(card_x, chart_titles, strict=True):
        draw_text(image, draw, (x, p(519)), label, chart_title)
    for x, key in zip(card_x, ("original", "english", "marathi"), strict=True):
        draw_plot(draw, (x + p(12), p(555), x + card_w - p(12), p(699)), distributions[key], tick_font)

    image.convert("RGB").save(OUTPUT, optimize=True)
    shutil.copy2(OUTPUT, REPO_ROOT / "images/example.png")
    print(f"Written: {OUTPUT}")


if __name__ == "__main__":
    render()
