# /// script
# dependencies = ["playwright"]
# ///
"""Export example.html and headline_figure.html to PNG.

Usage:
    uv run images/export_png.py
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def export(stem: str = "example"):
    html_path = Path(__file__).parent / f"{stem}.html"
    out_path = Path(__file__).parent / f"{stem}.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(device_scale_factor=2)
        await page.goto(html_path.resolve().as_uri())

        container = await page.query_selector(".container, #headline-figure")
        box = await container.bounding_box()
        width = 920 if "example" in stem else int(box["width"])
        await page.set_viewport_size({"width": width, "height": int(box["height"] + box["y"] * 2)})

        await page.screenshot(
            path=str(out_path),
            omit_background=True,
            full_page=True,
        )
        await browser.close()

    print(f"Written: {out_path}")


for name in ["example", "headline_figure"]:
    asyncio.run(export(name))
