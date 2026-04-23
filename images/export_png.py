# /// script
# dependencies = ["playwright"]
# ///
"""Export example.html to a transparent PNG.

Usage:
    uv run figures/export_png.py
"""

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright


async def export():
    html_path = Path(__file__).parent / "example.html"
    out_path = Path(__file__).parent / "example.png"

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html_path.resolve().as_uri())

        # Size the viewport to the container width; height auto-fits
        await page.set_viewport_size({"width": 920, "height": 100})
        container = await page.query_selector(".container")
        box = await container.bounding_box()
        await page.set_viewport_size({"width": 920, "height": int(box["height"] + box["y"] * 2)})

        await page.screenshot(
            path=str(out_path),
            omit_background=True,
            full_page=True,
        )
        await browser.close()

    print(f"Written: {out_path}")


asyncio.run(export())
