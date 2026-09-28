# /// script
# dependencies = ["playwright"]
# ///
"""Render headline_figure.html to headline_figure.png for the repository README.

The HTML is the source of truth; it embeds headline_curve_overlay.png, which
make_headline_curve.py writes. Rerun both after the underlying numbers change.

Usage:
    uv run images/export_png.py
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).parent
HTML = HERE / "headline_figure.html"
PNG = HERE / "headline_figure.png"


async def export() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=2)
        await page.goto(HTML.resolve().as_uri())

        # Size the viewport to the figure itself so the screenshot has no margin.
        box = await (await page.query_selector("#headline-figure")).bounding_box()
        await page.set_viewport_size({"width": int(box["width"]), "height": int(box["height"] + box["y"] * 2)})

        await page.screenshot(path=str(PNG), omit_background=True, full_page=True)
        await browser.close()

    print(f"Written: {PNG}")


asyncio.run(export())
