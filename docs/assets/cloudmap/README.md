# cloudmap assets

Data for the interactive version of paper Figure 6 (the descriptor-space cloud map).
Built by `_public/tools/site_assets/cloudmap/build_points.py` (+ `gpubox_cloudmap.py`),
which imports the figure's own code (`plot_cloud_map.build_sm_panels`), so the coordinates
are exactly those plotted in the paper.

## points.json

Top level:

| key | meaning |
|---|---|
| `figure`, `source`, `projection` | provenance strings |
| `features` | the 7 descriptor names |
| `colours` | `off` `#D55E00`, `on` `#0072B2`, `real` `#999999` (as in the figure) |
| `variants` | model variant names used in `fills[].variant` |
| `rows`, `cols` | stitch order (rows) and loss-size order (columns) |
| `panels` | 9 panels, row-major (french_knot 128/256/512, satin ..., silk_purl ...) |

Each panel:

| key | meaning |
|---|---|
| `stitch`, `stitch_label`, `size`, `hole_px` | e.g. `satin`, `Satin`, `256x256`, `256` |
| `title`, `xlabel`, `ylabel` | as in the figure (`256 px hole`, `PC1`, `Satin PC2`) |
| `xlim`, `ylim`, `ticks`, `tick_labels` | shared [-10, 10] view; edge tick labels blank as in the figure |
| `var_explained_pct` | % variance of the two projection axes (not shown in the figure) |
| `real_radius` | dashed ring radius (mean real self-distance), centred at the origin (real centre) |
| `com_off`, `com_on` | [x, y] centres of the off / on circles |
| `mean_D_off`, `mean_D_on`, `delta_pct` | circle radii = mean Mahalanobis D; Δ = (off − on)/off × 100 |
| `strip` | the strings printed in the panel strip, e.g. `{"off":"4.91","on":"2.64","delta":"Δ+46%"}` |
| `n_real`, `n_off`, `n_on` | dot counts |
| `real` | `[[x, y], ...]` real reference crops (grey cloud) |
| `real_sel` | selectable real dots (1 per panel, 3 per stitch): `{i, D, id, window_px, img, nrm, bbox}` — `i` indexes `real`, `id` is the held-out tile, `window_px` = [x0, y0, side] on the 1024 px tile |
| `fills` | all s=0 and s=1 fills: `{x, y, s, D, variant, id}` — `s` 0 = LoRA off, 1 = LoRA on; `id` = tile stem |

Selectable fills (10 per panel: 5 off + 5 on, as 5 same-tile/same-variant off/on pairs) carry
extra keys: `img`, `nrm`, `bbox`, `file` (source fill name) and `pair` (index in `fills` of the
other strength for the same tile, size and variant).

Notes:
- `D` is the full 7-D Mahalanobis distance to the size-matched real cloud; it equals the
  distance from the origin in whitened 7-D space, not in the 2-D projection.
- Coordinates are rounded to 3 dp. Some off dots fall outside the ±10 view (the figure clips
  them too); all selectable dots are inside it.

## img/

`<key>.webp` = the whole 1024 px tile in colour, at 384 × 384. `<key>_n.webp` = the normal map of
the fill region (or real crop window) only, enlarged to 256 × 256 (rgb = (n + 1)/2 × 255; flat ≈
lavender blue). This region is exactly what the descriptors are computed on. WebP q80. `bbox` =
[x0, y0, x1, y1] of the region in tile-normalised coordinates (0 to 1); all fill regions are
axis-aligned squares, so the box is the exact outline.
