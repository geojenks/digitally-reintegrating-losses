"""
paint_stitch_masks.py — brush-paint three disjoint stitch-type masks on one image,
for staged reintegration (fill region 1 with embfnchknt, then region 2 with embstn,
then region 3 with embslkprl).

The three layers are painted in distinct colours for visibility but saved as plain
binary masks (255 = fill this region with that stitch). Layers are kept disjoint:
painting one region erases any overlap from the other two, so a pixel belongs to at
most one stitch (clean staged compositing downstream).

Controls
--------
  1 / 2 / 3      select stitch:  1=french knot (blue)  2=satin (green)  3=silk purl (red)
  left-drag      paint the active layer
  right-drag     erase the active layer
  [  /  ]        brush smaller / larger
  c              clear the active layer        x = clear all
  s              save masks (+ preview)        q = quit
Saved (to --out, default beside the image):
  <stem>__french_knot.png  __satin.png  __silk_purl.png   (binary, native res)
  <stem>__paint_preview.png   (the colour overlay, for reference)

Usage
-----
venv\\Scripts\\python paint_stitch_masks.py --image path\\to\\piece.tif --out masks_out
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# (folder name, trigger token, RGB colour)
STITCHES = [
    ("french_knot", "embfnchknt", (31, 119, 180)),
    ("satin",       "embstn",     (44, 160, 44)),
    ("silk_purl",   "embslkprl",  (214, 39, 40)),
]
ALPHA = 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image to paint on")
    ap.add_argument("--out", default=None, help="output dir (default: beside the image)")
    ap.add_argument("--brush", type=int, default=0, help="initial brush radius px (0=auto)")
    args = ap.parse_args()

    img_path = Path(args.image)
    img = np.asarray(Image.open(img_path).convert("RGB"))
    H, W = img.shape[:2]
    out = Path(args.out) if args.out else img_path.parent
    out.mkdir(parents=True, exist_ok=True)

    # pre-load any existing masks for this image so you can add to / edit them
    layers = []
    loaded = []
    for name, _, _ in STITCHES:
        mp = out / f"{img_path.stem}__{name}.png"
        if mp.exists():
            m = np.asarray(Image.open(mp).convert("L").resize((W, H), Image.NEAREST))
            layers.append(((m > 127).astype(np.uint8) * 255))
            loaded.append(name)
        else:
            layers.append(np.zeros((H, W), np.uint8))
    if loaded:
        print(f"loaded existing masks: {', '.join(loaded)} (edit and re-save with s)")
    st = {"active": 0, "r": args.brush or max(8, H // 60), "pressed": False, "erase": False}

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.subplots_adjust(0, 0, 1, 0.93)
    ax.imshow(img)
    ov = ax.imshow(np.zeros((H, W, 4), np.uint8))
    ax.set_axis_off()

    def title():
        name, tok, _ = STITCHES[st["active"]]
        ax.set_title(f"active: {name} ({tok})   brush={st['r']}px   "
                     f"{'ERASE' if st['erase'] else 'paint'}   "
                     f"[1/2/3 stitch  L=paint R=erase [ ] size  c clear  x all  s save  q quit]",
                     fontsize=10, color=tuple(c / 255 for c in STITCHES[st["active"]][2]))

    def refresh():
        rgba = np.zeros((H, W, 4), np.uint8)
        for lay, (_, _, col) in zip(layers, STITCHES):
            m = lay > 0
            rgba[m] = (*col, ALPHA)
        ov.set_data(rgba)
        title()
        fig.canvas.draw_idle()

    def stamp(cx, cy, paint):
        r = st["r"]
        x0, x1 = max(0, cx - r), min(W, cx + r + 1)
        y0, y1 = max(0, cy - r), min(H, cy + r + 1)
        if x0 >= x1 or y0 >= y1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        a = st["active"]
        if paint:
            layers[a][y0:y1, x0:x1][disc] = 255
            for i in range(len(layers)):          # keep layers disjoint
                if i != a:
                    layers[i][y0:y1, x0:x1][disc] = 0
        else:
            layers[a][y0:y1, x0:x1][disc] = 0

    def on_press(e):
        if e.inaxes != ax or e.xdata is None:
            return
        st["pressed"] = True
        st["erase"] = (e.button == 3)
        stamp(int(e.xdata), int(e.ydata), paint=not st["erase"])
        refresh()

    def on_move(e):
        if not st["pressed"] or e.inaxes != ax or e.xdata is None:
            return
        stamp(int(e.xdata), int(e.ydata), paint=not st["erase"])
        refresh()

    def on_release(_):
        st["pressed"] = False

    def save():
        stem = img_path.stem
        for lay, (name, _, _) in zip(layers, STITCHES):
            Image.fromarray(lay).save(out / f"{stem}__{name}.png")
        prev = img.copy()
        for lay, (_, _, col) in zip(layers, STITCHES):
            m = lay > 0
            prev[m] = (0.45 * prev[m] + 0.55 * np.array(col)).astype(np.uint8)
        Image.fromarray(prev).save(out / f"{stem}__paint_preview.png")
        cov = [f"{name} {100*(lay>0).mean():.1f}%" for lay, (name, _, _) in zip(layers, STITCHES)]
        print(f"saved 3 masks + preview -> {out}   coverage: {', '.join(cov)}")

    def on_key(e):
        if e.key in ("1", "2", "3"):
            st["active"] = int(e.key) - 1; title(); fig.canvas.draw_idle()
        elif e.key == "]":
            st["r"] = min(400, st["r"] + 4); title(); fig.canvas.draw_idle()
        elif e.key == "[":
            st["r"] = max(2, st["r"] - 4); title(); fig.canvas.draw_idle()
        elif e.key == "c":
            layers[st["active"]][:] = 0; refresh()
        elif e.key == "x":
            for lay in layers:
                lay[:] = 0
            refresh()
        elif e.key == "s":
            save()
        elif e.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()


if __name__ == "__main__":
    main()
