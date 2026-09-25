"""Build the Marigold normal-estimation comparison assets for the project page.

Output: _public/docs/assets/marigold/  (384x384 WebP panels, angular-error maps,
colourbar.png, manifest.json)

Sources
-------
GPU box: D:\\marigold_finetune\\high_variance_eval\\
  * <tile_id>.png - 4-panel strip [colour | GT normal | public Marigold
    baseline | our fine-tune], each 768x768, 4 px grey gap (3084x768).
    These are the plain strips; the overlay\\ subfolder holds annotated copies
    and is NOT used.
  * results.csv - tile_id, embroidery, variance, base_mae, ft_mae, improvement
Produced by "Marigold training\\evaluate_high_variance_tiles.py", which encodes
normals as uint8((n + 1) * 127.5) (truncating), so flat = (127,127,255).
Its MAE is computed on the float predictions before 8-bit encoding.

Copy the CSV and the chosen strips to a local folder and pass --src, e.g.
    ssh gpubox "tar -cf - -C D:/marigold_finetune/high_variance_eval results.csv <tile>.png ..." | tar -xf - -C <SRC>

Tile choice: the paper's three tiles, then the highest-variance tiles per
embroidery, 3 each for e1-e4, e6, e7 and 2 for e5 (20 in total).
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "_public" / "docs" / "assets" / "marigold"

PAPER = ["e2_r019_t05_05", "e7_r023_t10_02", "e1_r022_t09_02"]
QUOTA = {"e1": 3, "e2": 3, "e3": 3, "e4": 3, "e5": 2, "e6": 3, "e7": 3}
PANEL, GAP, SIZE, QUALITY = 768, 4, 384, 80
VMAX = 60.0
CMAP = colormaps["magma"]
NAMES = ["colour", "gt", "base", "ft"]


def select(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["embroidery"]].append(r)
    sel = list(PAPER)
    for e in sorted(by):
        have = sum(t.startswith(e + "_") for t in sel)
        for r in sorted(by[e], key=lambda r: -float(r["variance"])):
            if have >= QUOTA[e]:
                break
            if r["tile_id"] not in sel:
                sel.append(r["tile_id"])
                have += 1
    return sel


def decode(u8, half=True):
    """uint8 RGB -> unit normals. half=True undoes the encoder's truncation."""
    n = (u8.astype(np.float64) + (0.5 if half else 0.0)) / 127.5 - 1.0
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def ang(a, b):
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def err_image(deg):
    rgb = CMAP(np.clip(deg / VMAX, 0, 1))[..., :3]
    im = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))
    return im.resize((SIZE, SIZE), Image.LANCZOS)


def colourbar(path, w=512, h=20):
    x = np.linspace(0, 1, w)[None, :].repeat(h, 0)
    Image.fromarray((CMAP(x)[..., :3] * 255 + 0.5).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.src / "results.csv", encoding="utf-8")))
    idx = {r["tile_id"]: r for r in rows}
    manifest, check = [], []
    for t in select(rows):
        r = idx[t]
        strip = np.array(Image.open(args.src / f"{t}.png").convert("RGB"))
        assert strip.shape == (PANEL, 4 * PANEL + 3 * GAP, 3), strip.shape
        panels = [strip[:, i * (PANEL + GAP): i * (PANEL + GAP) + PANEL] for i in range(4)]
        files = {}
        for name, p in zip(NAMES, panels):
            files[name] = f"{t}_{name}.webp"
            Image.fromarray(p).resize((SIZE, SIZE), Image.LANCZOS).save(
                OUT / files[name], "WEBP", quality=QUALITY, method=6)
        gt = decode(panels[1])
        stats = {}
        for name, p in (("base", panels[2]), ("ft", panels[3])):
            d = ang(decode(p), gt)
            stats[name] = float(d.mean())
            stats[name + "_nohalf"] = float(ang(decode(p, False), decode(panels[1], False)).mean())
            files[name + "_err"] = f"{t}_{name}_err.webp"
            err_image(d).save(OUT / files[name + "_err"], "WEBP", quality=QUALITY, method=6)
        check.append((t, float(r["base_mae"]), stats["base"], stats["base_nohalf"],
                      float(r["ft_mae"]), stats["ft"], stats["ft_nohalf"]))
        manifest.append({"tile": t, "embroidery": r["embroidery"],
                         "variance": round(float(r["variance"]), 5),
                         "base_mae": round(float(r["base_mae"]), 2),
                         "ft_mae": round(float(r["ft_mae"]), 2),
                         "paper": t in PAPER, "files": files})
    colourbar(OUT / "colourbar.png")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"{'tile':16s} {'csvB':>6s} {'8bB':>6s} {'8bB-':>6s} {'csvF':>6s} {'8bF':>6s} {'8bF-':>6s}")
    for c in check:
        print(f"{c[0]:16s} " + " ".join(f"{v:6.2f}" for v in c[1:]))
    a = np.array([c[1:] for c in check])
    for lab, i, j in (("base", 0, 1), ("base (no half-step)", 0, 2), ("ft", 3, 4), ("ft (no half-step)", 3, 5)):
        d = a[:, j] - a[:, i]
        print(f"{lab:22s} 8-bit minus CSV: mean {d.mean():+.3f}, max |d| {abs(d).max():.3f} deg")


if __name__ == "__main__":
    main()
