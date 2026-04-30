#!/usr/bin/env python
# /// script
# dependencies = ["geopandas", "matplotlib", "pandas", "multilingual-gsm-symbolic"]
# [tool.uv.sources]
# multilingual-gsm-symbolic = { path = "../..", editable = true }
# ///
"""Create a world map colored by language creation method.

Uses Natural Earth for the world layer and geoBoundaries for Ukraine/Russia,
so Crimea is shown with Ukraine.
Ukraine is plotted like any other country (no special highlighting).
"""

import tomllib
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_hex

from multilingual_gsm_symbolic.load_data import available_languages

GEObOUNDARIES_URLS = {
    "UKR": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/UKR/ADM0/geoBoundaries-UKR-ADM0_simplified.geojson",
    "RUS": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RUS/ADM0/geoBoundaries-RUS-ADM0_simplified.geojson",
}

LANGUAGE_COUNTRIES = {
    "eng": ["USA", "GBR", "CAN", "AUS", "NZL", "IRL", "ZAF"],
    "dan": ["DNK"],
    "deu": ["DEU", "AUT", "CHE"],
    "nob": ["NOR"],
    "isl": ["ISL"],
}

NO_COVERAGE_COLOR = "#E0E0E0"

CATEGORY_LABELS = {
    "original": "Original",
    "human_validated": "Human-translated, localized, validated",
    "machine_translated": "Machine-translated and machine-validated",
}


def get_creation_method(lang: str, templates_dir: Path) -> str:
    """Determine the creation method for a language based on its templates."""
    lang_dir = templates_dir / lang / "symbolic"
    if not lang_dir.exists():
        return "none"

    first_template = sorted(lang_dir.glob("*.toml"))[0]
    with first_template.open("rb") as f:
        data = tomllib.load(f)
    creation = data.get("creation", "")

    if "derived from GSM-Symbolic" in creation:
        return "original"
    if "human" in creation.lower():
        return "human_validated"
    return "machine_translated"


def build_category_colors(methods: list[str]) -> dict[str, str]:
    """Assign colors from derived categories with an ordered quality palette.

    Colors still come from the parsed creation categories, but are mapped so that:
    - original = blue
    - human_validated = green
    - machine_translated = orange/red
    """
    colors: dict[str, str] = {}
    present = set(methods)

    if "original" in present:
        colors["original"] = to_hex(plt.get_cmap("Blues")(0.75))
    if "human_validated" in present:
        colors["human_validated"] = to_hex(plt.get_cmap("RdYlGn")(0.92))
    if "machine_translated" in present:
        colors["machine_translated"] = to_hex(plt.get_cmap("RdYlGn")(0.22))

    return colors


def replace_country_geometry(world: gpd.GeoDataFrame, iso_a3: str, source_url: str) -> gpd.GeoDataFrame:
    """Replace a country's geometry using geoBoundaries."""
    replacement = gpd.read_file(source_url).to_crs(world.crs).copy()
    replacement["ISO_A3"] = iso_a3

    # Rebuild the replacement frame in one step to avoid pandas fragmentation warnings.
    replacement_data = {
        col: replacement[col] if col in replacement.columns else pd.Series([None] * len(replacement))
        for col in world.columns
        if col != "geometry"
    }
    replacement_aligned = pd.DataFrame(replacement_data, index=replacement.index)
    replacement_aligned["geometry"] = replacement.geometry
    replacement_aligned = gpd.GeoDataFrame(replacement_aligned, geometry="geometry", crs=world.crs)

    world_without_country = world[world["ISO_A3"] != iso_a3].copy()
    combined = pd.concat([world_without_country, replacement_aligned], ignore_index=True).copy()
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=world.crs)


def load_world() -> gpd.GeoDataFrame:
    """Load world countries and replace UKR/RUS with geoBoundaries versions."""
    try:
        print("Downloading Natural Earth 50m countries...")
        world = gpd.read_file("https://naturalearth.s3.amazonaws.com/50m_cultural/ne_50m_admin_0_countries.zip")
        print(f"Loaded {len(world)} countries from Natural Earth 50m")
    except Exception as e:
        print(f"Failed to load Natural Earth 50m data: {e}")
        print("Trying to load Natural Earth 110m instead...")
        world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")
        print(f"Loaded {len(world)} countries from Natural Earth 110m")

    print("Replacing Ukraine/Russia geometries with geoBoundaries...")
    world = replace_country_geometry(world, "UKR", GEObOUNDARIES_URLS["UKR"])
    world = replace_country_geometry(world, "RUS", GEObOUNDARIES_URLS["RUS"])
    print("Replaced UKR and RUS geometries from geoBoundaries")
    return world


def main() -> None:
    templates_dir = Path("src/multilingual_gsm_symbolic/data/templates")
    langs = list(available_languages().keys())
    print(f"Found languages: {langs}")

    lang_methods = {lang: get_creation_method(lang, templates_dir) for lang in langs}
    category_colors = build_category_colors(list(lang_methods.values()))

    world = load_world()
    world["color"] = NO_COVERAGE_COLOR
    world["method"] = "none"

    for lang in langs:
        method = lang_methods[lang]
        color = category_colors[method]
        for country_iso in LANGUAGE_COUNTRIES.get(lang, []):
            mask = world["ISO_A3"] == country_iso
            if mask.any():
                world.loc[mask, "color"] = color
                world.loc[mask, "method"] = method
                print(f"  {lang}: {CATEGORY_LABELS[method]} -> {country_iso}")

    fig, ax = plt.subplots(figsize=(16, 10))

    for method, color in category_colors.items():
        subset = world[world["method"] == method]
        if not subset.empty:
            subset.plot(
                ax=ax,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.8,
            )

    rest_of_world = world[world["method"] == "none"]
    if not rest_of_world.empty:
        rest_of_world.plot(
            ax=ax,
            facecolor=NO_COVERAGE_COLOR,
            edgecolor="#CCCCCC",
            linewidth=0.3,
            alpha=0.5,
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Language Coverage by Creation Method", fontsize=14, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        *[
            plt.Rectangle((0, 0), 1, 1, facecolor=color, label=CATEGORY_LABELS[method])
            for method, color in category_colors.items()
        ],
        plt.Rectangle((0, 0), 1, 1, facecolor=NO_COVERAGE_COLOR, label="No coverage"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, fancybox=True, fontsize=9)

    output_path = Path("images/language_coverage_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved map to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
