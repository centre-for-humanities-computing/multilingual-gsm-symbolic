# Full comparison layouts

## Two-page ICLR version (recommended)

`full_comparisons_two_pages.tex` puts each 48-model comparison on its own page,
with two columns per image. It uses the unmodified official
[ICLR 2026 style](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.sty),
included here, with its 5.5-inch text width and 9-inch text height. No geometry
or caption-size overrides are applied. Keep your manuscript's existing
year-specific style and copy only the figure blocks into it.

The new PNGs are `eng_vs_eng_metric_full_fullpage.png` and
`correction_comparison/isl_fullpage.png`. Both are generated alongside the
original compact PNGs by the command below. Full-page labels are 6.5 points;
complete reasoning suffixes fit on one line. Most row height is reserved for
curves. Full-page plots omit histogram bars and set the vertical scale from
the fitted curves, preventing tall histogram bins from flattening the curves.
Palettes, sampled data, and the fitted curves themselves are unchanged.

Compile from `paper/latex`:

```sh
tectonic full_comparisons_two_pages.tex --keep-logs --outdir ../artifacts/figures/model_grid
```

Acceptance: rendered and visually inspected both pages. Exactly two pages,
no clipped labels or overlapping panels, and no overfull/underfull boxes.
Full slugs and broad curve differences are readable. Fine differences between
nearly coincident, narrow English curves still require zoom; the shared 0–100%
axis is retained rather than silently magnifying individual distributions.

## Preserved one-page version

`full_comparisons_page.tex`, its compact PNGs, and its original PDF proof remain
available. This earlier proof uses custom 7-inch text width; it is **not** the
ICLR-format acceptance proof. Use the two-page file above for ICLR.

Regenerate the two complete PNGs from the repository root:

```sh
uv run --script paper/scripts/render_full_comparisons.py
```

Compile the acceptance proof from `paper/latex`:

```sh
tectonic full_comparisons_page.tex --keep-logs --outdir ../artifacts/figures/model_grid
```

The PNGs contain two columns each, with all model labels retained. English uses
blue/orange; Icelandic uses purple/teal. Solid/dashed lines distinguish the two
conditions in each comparison. Histograms are subdued so fitted curves remain
visible. Sampling and fitted distributions are unchanged.

For the manuscript, copy the figure environment from the proof and adapt the
image paths. Use `figure*` for a two-column manuscript. Do not copy the proof's
document class or margins into the manuscript.

## Acceptance review (2026-09-11)

- Data: upstream paper-content commit `eaf3d06e`, including the newly completed
  historical Icelandic results. Both comparisons contain 48 paired models.
- Proof: US Letter, 7-inch text width, 0.75-inch margins, two separate captions.
  The actual manuscript class is not available in this checkout, so this is an
  explicit layout approximation rather than verification of that template.
- Compiled with Tectonic and rendered with Poppler. The PDF is exactly one page;
  the LaTeX log contains no overfull/underfull box warnings. Inspected the rendered
  page for labels, clipping, legend separation, and curve readability.
- Fit and completeness: PASS. All 96 panels fit; full slugs and reasoning suffixes
  are retained. There are no embedded figure headings.
- Easy reading at paper size: FAIL. Labels are approximately 6.1 points after
  inclusion. Broad Icelandic shifts are visible, but narrow English distributions
  near 100% and small paired differences require magnification. High PNG resolution
  preserves detail when zoomed but cannot fix the physical-size limitation.

The four-column page is suitable as a compact overview, not as an easily readable
full distribution comparison. To satisfy that stronger gate, allow more pages
and larger panels (and consider explicitly labelled local x-axis ranges for very
narrow distributions). The current PNGs retain the common 0–100% axis and all data.
