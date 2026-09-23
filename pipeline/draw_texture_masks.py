"""
draw_texture_masks.py — quick freehand masks marking the VALID texture region of
each reference tile, so the descriptor metric only samples sub-crops that fall
entirely inside real stitching (no background fabric, frayed edges, or gaps).

Workflow per image:
  * lasso-drag around the good texture; release to add that region to the mask.
  * draw as many lassos as you like (they union). Hold and drag for freehand.
  * keys:  [enter] or n = save mask + next      a = select all (whole tile valid)
           r = reset this image                  b = back (previous image)
           s = skip (no mask written)            q = quit (saved so far; resumable)
Masks are written as <stem>.png (uint8, 255 = valid) into --out, at the image's
native resolution so they align with the matching normal-map .npy.

Resumable: images that already have a mask in --out are skipped unless --overwrite.

Usage
-----
venv\\Scripts\\python draw_texture_masks.py ^
    --images "data_stitches\\holdouts-synthetic_damage\\*\\images\\*.tif" ^
    --out    references\\texture_masks
# then (next step) point the metric's crop sampler at --maskdir references\\texture_masks
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="glob of source images (RGB)")
    ap.add_argument("--out", required=True, help="output dir for <stem>.png masks")
    ap.add_argument("--overwrite", action="store_true", help="redo images that already have a mask")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in sorted(glob.glob(args.images))]
    if not paths:
        raise SystemExit(f"No images match {args.images}")
    todo = [p for p in paths if args.overwrite or not (out / f"{p.stem}.png").exists()]
    print(f"{len(paths)} images, {len(todo)} to mask (rest already done).")
    if not todo:
        return

    state = {"i": 0, "mask": None, "img": None, "shape": None}

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.subplots_adjust(0, 0, 1, 0.94)

    def load(i):
        p = todo[i]
        img = np.asarray(Image.open(p).convert("RGB"))
        state.update(img=img, shape=img.shape[:2], mask=np.zeros(img.shape[:2], np.uint8))
        ax.clear()
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(f"[{i+1}/{len(todo)}] {p.name}   "
                     f"lasso=add  a=all  n=save+next  r=reset  b=back  s=skip  q=quit",
                     fontsize=10)
        fig.canvas.draw_idle()

    def redraw_overlay():
        ax.imshow(state["img"])
        if state["mask"].any():
            ov = np.zeros((*state["shape"], 4), np.uint8)
            ov[state["mask"] > 0] = (0, 200, 0, 90)
            ax.imshow(ov)
        ax.set_axis_off()
        fig.canvas.draw_idle()

    def on_select(verts):
        # rasterise the lasso polygon and union it into the mask
        poly = [(float(x), float(y)) for x, y in verts]
        if len(poly) < 3:
            return
        m = Image.new("L", (state["shape"][1], state["shape"][0]), 0)
        ImageDraw.Draw(m).polygon(poly, fill=255)
        state["mask"] = np.maximum(state["mask"], np.asarray(m))
        redraw_overlay()

    lasso = LassoSelector(ax, on_select, button=1)

    def save_current():
        p = todo[state["i"]]
        Image.fromarray(state["mask"]).save(out / f"{p.stem}.png")
        cov = 100.0 * (state["mask"] > 0).mean()
        print(f"  saved {p.stem}.png  ({cov:.0f}% valid)")

    def go(i):
        if i >= len(todo):
            print("All done.")
            plt.close(fig)
            return
        if i < 0:
            i = 0
        state["i"] = i
        load(i)

    def on_key(event):
        if event.key in ("enter", "n"):
            save_current(); go(state["i"] + 1)
        elif event.key == "a":
            state["mask"] = np.full(state["shape"], 255, np.uint8); redraw_overlay()
        elif event.key == "r":
            state["mask"] = np.zeros(state["shape"], np.uint8); redraw_overlay()
        elif event.key == "b":
            go(state["i"] - 1)
        elif event.key == "s":
            print(f"  skipped {todo[state['i']].name}"); go(state["i"] + 1)
        elif event.key == "q":
            print("Quit (progress saved)."); plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    go(0)
    plt.show()


if __name__ == "__main__":
    main()
