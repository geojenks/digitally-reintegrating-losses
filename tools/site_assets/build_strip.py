"""Build the step-by-step loss-strip assets (paper Figure 8) for the project page.

Output: _public/docs/assets/strip/  (900px-wide WebP, quality 85, manifest.json)

Sources (all 1024x1024, paths relative to the main repo root)
-------------------------------------------------------------
1 photograph        _public/data/masks/castle.png
2 mask overlay      masks_out/castle__paint_preview.png
  binary masks      _public/data/masks/castle__{french_knot,satin,silk_purl}.png
3 procedural init   picker_castle/_paste_preview/castle/paste_preview/
                      stage1_french_knot_pasted.png  (French knot regions filled)
                      stage2_satin_pasted.png        (+ satin regions)
                      stage3_silk_purl_pasted.png    (+ silk purl = full init)
  The stages are cumulative: each only changes pixels inside its own
  stitch's mask (checked when this was built).
4 final             _public/media/castle_reintegrated.png

The mask overlays are transparent everywhere except their stitch's region,
where they carry that stitch's colour at ~55% alpha. The colours match the
hues of the paint preview (French knot blue, satin green, silk purl red), so
stacking the three over step1 closely reproduces step2.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "_public" / "docs" / "assets" / "strip"
WIDTH, QUALITY = 900, 85
ALPHA = 0.55
PASTE = ROOT / "picker_castle" / "_paste_preview" / "castle" / "paste_preview"
STITCHES = ["french_knot", "satin", "silk_purl"]
LABELS = {"french_knot": "French knot", "satin": "Satin stitch", "silk_purl": "Silk purl"}
COLOURS = {"french_knot": (30, 118, 180), "satin": (43, 160, 43), "silk_purl": (213, 37, 35)}

STEPS = [
    ("step1.webp", ROOT / "_public/data/masks/castle.png",
     "Photograph: the design survives, but the stitches are lost"),
    ("step2.webp", ROOT / "masks_out/castle__paint_preview.png",
     "Mask each region by stitch type"),
    ("step3.webp", PASTE / "stage3_silk_purl_pasted.png",
     "Fill each region with a procedural stand-in of its stitch"),
    ("step4.webp", ROOT / "_public/media/castle_reintegrated.png",
     "Add noise and denoise with the stitch LoRA"),
]
INITS = [
    ("init_stage1.webp", PASTE / "stage1_french_knot_pasted.png", "french_knot",
     "Procedural fill: French knot regions"),
    ("init_stage2.webp", PASTE / "stage2_satin_pasted.png", "satin",
     "Procedural fill: French knot and satin regions"),
]


def save(im, name):
    h = round(im.height * WIDTH / im.width)
    im.resize((WIDTH, h), Image.LANCZOS).save(OUT / name, "WEBP", quality=QUALITY, method=6)
    return [WIDTH, h]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = {"width": WIDTH, "steps": [], "masks": [], "inits": []}
    for i, (name, src, cap) in enumerate(STEPS, 1):
        size = save(Image.open(src).convert("RGB"), name)
        man["steps"].append({"step": i, "file": name, "caption": cap, "size": size})
    for s in STITCHES:
        m = np.array(Image.open(ROOT / f"_public/data/masks/castle__{s}.png").convert("L")) > 127
        rgba = np.zeros(m.shape + (4,), np.uint8)
        rgba[m, :3] = COLOURS[s]
        rgba[m, 3] = round(ALPHA * 255)
        name = f"mask_{s}.webp"
        im = Image.fromarray(rgba, "RGBA")
        h = round(im.height * WIDTH / im.width)
        im.resize((WIDTH, h), Image.LANCZOS).save(OUT / name, "WEBP", quality=QUALITY,
                                                   method=6, exact=True)
        man["masks"].append({"stitch": s, "label": LABELS[s], "file": name,
                             "colour": "#%02x%02x%02x" % COLOURS[s], "alpha": ALPHA,
                             "area_fraction": round(float(m.mean()), 4)})
    for name, src, s, cap in INITS:
        save(Image.open(src).convert("RGB"), name)
        man["inits"].append({"file": name, "after_stitch": s, "caption": cap})
    man["inits"].append({"file": "step3.webp", "after_stitch": "silk_purl",
                         "caption": "Procedural fill: all regions"})
    man["note"] = ("Masks are disjoint; overlays are transparent outside their region and "
                   "can be stacked over step1.webp in any order. init_stage1/2 and step3 "
                   "are the cumulative procedural paste, one stitch at a time.")
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
