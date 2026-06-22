# Family marker encoding

## Goal

Use the established model-family marker shapes in every paper plot instead of
using color to distinguish model families.

## Approach

- Keep `visualizegrid.py`'s existing `FAMILY_MARKERS` mapping as the canonical
  shapes.
- Render family series, points, and legends in a neutral color while preserving
  marker shapes.
- Add the same marker mapping to `transferfeatures.py` and apply it to its
  family series.
- Retain color when it encodes another dimension, such as language or split.

## Verification

Add a source-level test covering the marker mapping and the absence of family
color selection in family-encoded plot calls, then run that test.
