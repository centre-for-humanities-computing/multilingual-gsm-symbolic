#!/usr/bin/env python
# /// script
# dependencies = ["geopandas", "matplotlib", "pandas", "pycountry", "multilingual-gsm-symbolic"]
# [tool.uv.sources]
# multilingual-gsm-symbolic = { path = "../..", editable = true }
# ///
"""Create a world map colored by language creation method.

Uses Natural Earth for the world layer and geoBoundaries for Ukraine/Russia,
so Crimea is shown with Ukraine.
Ukraine is plotted like any other country (no special highlighting).

Countries are matched to languages from Unicode CLDR supplemental data rather
than a hand-written list, so the map stays in sync as languages are added:

- A language covers its CLDR *primary* territories, plus any territory where
  CLDR marks it official (including regional official status). Languages CLDR
  lists no primary territory for (regional languages without official status,
  e.g. Bavarian or Chhattisgarhi) fall back to their secondary territories.
  This keeps English out of the many countries where it is only an L2.
- A country is colored by the creation method of the covered language spoken by
  the largest share of its population, per CLDR territory data.
"""

import json
import re
import tomllib
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pycountry

from multilingual_gsm_symbolic.load_data import available_languages

GEObOUNDARIES_URLS = {
    "UKR": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/UKR/ADM0/geoBoundaries-UKR-ADM0_simplified.geojson",
    "RUS": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/RUS/ADM0/geoBoundaries-RUS-ADM0_simplified.geojson",
}

CLDR_TAG = "46.1.0"
CLDR_URLS = {
    name: f"https://raw.githubusercontent.com/unicode-org/cldr-json/{CLDR_TAG}/cldr-json/cldr-core/supplemental/{name}.json"
    for name in ("territoryInfo", "languageData")
}

# ISO 639-3 codes used here that CLDR spells with a macrolanguage / different code.
CLDR_ALIASES = {
    "azb": "az",  # South Azerbaijani
    "gaz": "om",  # West Central Oromo
    "kmr": "ku",  # Northern Kurdish
    "ktu": "kg",  # Kituba
    "npi": "ne",  # Nepali
    "ory": "or",  # Odia
    "pbu": "ps",  # Northern Pashto
    "pes": "fa",  # Western Persian
    "pnb": "pa",  # Western Panjabi
    "swh": "sw",  # Swahili
    "tgl": "fil",  # Tagalog / Filipino
    "uzn": "uz",  # Northern Uzbek
    "zlm": "ms",  # Malay
}

# Languages CLDR has no territory data for at all.
EXTRA_TERRITORIES = {
    "ctg": ["BD"],  # Chittagonian
    "vjk": ["IN"],  # Bajjika
}

NO_COVERAGE_COLOR = "#E3E3E3"

# Ordered quality ramp; see build_category_colors.
METHOD_COLORS = {
    "original": "#08519C",
    "human_validated": "#3182BD",
    "machine_translated": "#A8CDE5",
}

CATEGORY_LABELS = {
    "original": "Original",
    "human_validated": "Translated and human-validated",
    "machine_translated": "Machine-translated and machine-validated",
}


# "validated by a native speaker", "human validated", "manually corrected by ..." etc.
HUMAN_VALIDATED = re.compile(r"human|native speaker|fluent speaker|manually corrected", re.IGNORECASE)


def classify_creation(creation: str) -> str:
    """Classify a single template's `creation` string."""
    if "derived from GSM-Symbolic" in creation:
        return "original"
    if HUMAN_VALIDATED.search(creation):
        return "human_validated"
    return "machine_translated"


def get_creation_method(lang: str, templates_dir: Path) -> str:
    """Determine the creation method for a language from its active templates.

    Languages are mixed (e.g. Ukrainian has a couple of templates derived straight
    from GSM-Symbolic among otherwise human-validated ones), so the language takes
    the method of the majority of its templates rather than of the first one.
    """
    lang_dir = templates_dir / lang / "symbolic"
    if not lang_dir.exists():
        return "none"

    methods = Counter()
    for template in sorted(lang_dir.glob("*.toml")):
        with template.open("rb") as f:
            data = tomllib.load(f)
        if data.get("ignore"):
            continue
        methods[classify_creation(data.get("creation", ""))] += 1

    if not methods:
        return "none"
    return methods.most_common(1)[0][0]


def build_category_colors(methods: list[str]) -> dict[str, str]:
    """Assign colors from derived categories along an ordered quality ramp.

    Colors still come from the parsed creation categories, but are mapped onto a
    single-hue ramp so the map reads as a quality scale: no coverage (light grey)
    -> machine-translated (light blue, deliberately closer to "no coverage") ->
    human-validated -> original (two adjacent dark blues, close but distinct).
    """
    return {method: color for method, color in METHOD_COLORS.items() if method in set(methods)}


def _load_cldr(name: str) -> dict:
    print(f"Downloading CLDR {name}...")
    with urllib.request.urlopen(CLDR_URLS[name]) as response:
        return json.load(response)["supplemental"][name]


def cldr_code(lang: str) -> str:
    """Map a template language code (ISO 639-3, possibly suffixed) to its CLDR code."""
    base = lang.split("_")[0]
    if base in CLDR_ALIASES:
        return CLDR_ALIASES[base]
    entry = pycountry.languages.get(alpha_3=base)
    return (getattr(entry, "alpha_2", None) if entry else None) or base


def language_territories(langs: list[str]) -> dict[str, dict[str, float]]:
    """Map each territory to its covered languages and their population share.

    Returns:
        Mapping of CLDR territory code -> {language code: percent of population}.
    """
    territory_info = _load_cldr("territoryInfo")
    language_data = _load_cldr("languageData")

    primary: dict[str, set[str]] = defaultdict(set)
    secondary: dict[str, set[str]] = defaultdict(set)
    for key, entry in language_data.items():
        target = secondary if key.endswith("-alt-secondary") else primary
        target[key.split("-")[0]].update(entry.get("_territories", []))

    # Population share and official status per territory, merged across script variants.
    percent: dict[str, dict[str, float]] = defaultdict(dict)
    official: dict[str, set[str]] = defaultdict(set)
    for territory, info in territory_info.items():
        if len(territory) != 2:  # skip regions such as "001" (World)
            continue
        for key, entry in info.get("languagePopulation", {}).items():
            code = key.split("_")[0]
            share = float(entry.get("_populationPercent", 0))
            percent[territory][code] = max(percent[territory].get(code, 0.0), share)
            if entry.get("_officialStatus"):
                official[territory].add(code)

    covered: dict[str, dict[str, float]] = defaultdict(dict)
    for lang in langs:
        code = cldr_code(lang)
        territories = primary.get(code, set()) | {t for t, codes in official.items() if code in codes}
        territories = territories or secondary.get(code, set())
        territories = territories or set(EXTRA_TERRITORIES.get(lang.split("_")[0], []))
        if not territories:
            print(f"  warning: no territory found for {lang} (CLDR code {code})")
        for territory in territories:
            covered[territory][lang] = percent.get(territory, {}).get(code, 0.0)
    return covered


def replace_country_geometry(world: gpd.GeoDataFrame, iso_a3: str, source_url: str) -> gpd.GeoDataFrame:
    """Replace a country's geometry using geoBoundaries."""
    replacement = gpd.read_file(source_url).to_crs(world.crs).copy()
    replacement["iso_a2"] = ISO3_TO_ISO2.get(iso_a3, "")

    # Rebuild the replacement frame in one step to avoid pandas fragmentation warnings.
    replacement_data = {
        col: replacement[col] if col in replacement.columns else pd.Series([None] * len(replacement))
        for col in world.columns
        if col != "geometry"
    }
    replacement_aligned = pd.DataFrame(replacement_data, index=replacement.index)
    replacement_aligned["geometry"] = replacement.geometry
    replacement_aligned = gpd.GeoDataFrame(replacement_aligned, geometry="geometry", crs=world.crs)

    world_without_country = world[world["iso_a2"] != ISO3_TO_ISO2.get(iso_a3, "")].copy()
    combined = pd.concat([world_without_country, replacement_aligned], ignore_index=True).copy()
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=world.crs)


ISO3_TO_ISO2 = {"UKR": "UA", "RUS": "RU"}


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

    # ISO_A2 is "-99" for a handful of countries (e.g. France, Norway); ISO_A2_EH fills those in.
    iso_a2 = world["ISO_A2_EH"] if "ISO_A2_EH" in world.columns else world["ISO_A2"]
    world["iso_a2"] = iso_a2.where(iso_a2.isin([c.alpha_2 for c in pycountry.countries]), "")

    print("Replacing Ukraine/Russia geometries with geoBoundaries...")
    world = replace_country_geometry(world, "UKR", GEObOUNDARIES_URLS["UKR"])
    world = replace_country_geometry(world, "RUS", GEObOUNDARIES_URLS["RUS"])
    print("Replaced UKR and RUS geometries from geoBoundaries")
    return world


def main() -> None:
    templates_dir = Path("src/multilingual_gsm_symbolic/data/templates")
    langs = list(available_languages().keys())
    print(f"Found {len(langs)} languages: {langs}")

    lang_methods = {lang: get_creation_method(lang, templates_dir) for lang in langs}
    category_colors = build_category_colors(list(lang_methods.values()))

    covered = language_territories(langs)
    # Each country takes the method of the covered language most of its population speaks.
    territory_method = {
        territory: lang_methods[max(langs_pct.items(), key=lambda item: item[1])[0]]
        for territory, langs_pct in covered.items()
    }

    world = load_world()
    world["method"] = world["iso_a2"].map(territory_method).fillna("none")
    world["color"] = world["method"].map(category_colors).fillna(NO_COVERAGE_COLOR)

    matched = world[world["method"] != "none"]
    print(f"Colored {len(matched)} countries; {len(set(territory_method) - set(world['iso_a2']))} territories unmatched")
    for method, count in world["method"].value_counts().items():
        print(f"  {method}: {count} countries")

    fig, ax = plt.subplots(figsize=(16, 10))

    for method, color in category_colors.items():
        subset = world[world["method"] == method]
        if not subset.empty:
            subset.plot(
                ax=ax,
                facecolor=color,
                edgecolor="#3D3D3D",
                linewidth=0.4,
            )

    rest_of_world = world[world["method"] == "none"]
    if not rest_of_world.empty:
        rest_of_world.plot(
            ax=ax,
            facecolor=NO_COVERAGE_COLOR,
            edgecolor="#C4C4C4",
            linewidth=0.3,
        )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 90)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        *[
            plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="#3D3D3D", linewidth=0.4, label=CATEGORY_LABELS[method])
            for method, color in category_colors.items()
        ],
        plt.Rectangle((0, 0), 1, 1, facecolor=NO_COVERAGE_COLOR, edgecolor="#C4C4C4", linewidth=0.4, label="No coverage"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", frameon=True, fancybox=True, fontsize=9)

    output_path = Path("images/language_coverage_map.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved map to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
