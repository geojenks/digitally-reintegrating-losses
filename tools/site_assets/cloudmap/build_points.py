r"""Local half of the interactive cloud-map build (paper Figure 6, fig:cloudmap).

Rebuilds the exact 2-D coordinates of the paper's cloud map (cloud_map_grid,
made by `plot_cloud_map.py --all_stitches --root _score_out_holdout_full
--font_scale 2.2`) by importing plot_cloud_map.build_sm_panels from the main
repo, then writes the web assets under _public/docs/assets/cloudmap/.

Sources
-------
  main repo      C:\Users\geo\Documents\Projects\Reintegrating Losses
  plot code      <main>\plot_cloud_map.py  (build_sm_panels: per-panel whitening by the
                 size-matched real covariance, then SVD of the pooled s=0/s=1 fills)
  scores         <main>\_score_out_holdout_full\<stitch>\scores_sm_<variant>.csv
  real clouds    <main>\_score_out_holdout_full\<stitch>\refcrops_<stitch>.npz
  images         on the gpubox, via gpubox_cloudmap.py (see its docstring):
                 D:\reintegration_sweep_out\<stitch>\<variant>\<size>\<stem>.{png,npy,mask.png}
                 D:\Projects\reintegrating_losses_paper\data_stitches\holdouts-synthetic_damage\<stitch>\images\<stem>.{tif,npy}

Steps (WORK defaults to a scratch folder; nothing on the gpubox is modified)
-----
  1. gpubox: copy gpubox_cloudmap.py to D:\_cloudmap_tmp, copy distance_to_real.py,
     classical_surface_descriptors.py, texture_fidelity.py from the repo into
     D:\_cloudmap_tmp\code, run `python -B gpubox_cloudmap.py reproduce reproduce.json`,
     scp reproduce.json into WORK.  (Reproduces refcrops_<stitch>.npz bit-exactly
     and records the tile + window of every real crop.)
  2. local:  python build_points.py select     -> WORK\crop_jobs.json
  3. gpubox: python -B gpubox_cloudmap.py crop crop_jobs.json crops ; scp crops\ back into WORK\crops
  4. local:  python build_points.py final      -> assets/cloudmap/points.json, img/*.webp,
                                                  WORK\verify_grid.png (re-render from the JSON)
  5. delete D:\_cloudmap_tmp.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

MAIN = r"C:\Users\geo\Documents\Projects\Reintegrating Losses"
ROOT = os.path.join(MAIN, "_score_out_holdout_full")
OUT = os.path.join(MAIN, "_public", "docs", "assets", "cloudmap")
WORK_DEFAULT = (r"C:\Users\geo\AppData\Local\Temp\claude\C--Users-geo-Documents-Projects-"
                r"Reintegrating-Losses\305ea9c4-270b-49d1-8538-10278601fae3\scratchpad\cloudmap\work")
SWEEP = r"D:\reintegration_sweep_out"
GB_REPO = r"D:\Projects\reintegrating_losses_paper"

sys.path.insert(0, MAIN)
import plot_cloud_map as pcm  # noqa: E402  (the figure's own code)

STITCHES = ["french_knot", "satin", "silk_purl"]
ABBR = {"french_knot": "fk", "satin": "sa", "silk_purl": "sp"}
LIM = 10.0            # the figure's --lim default (shared [-10, 10] view)


def r3(v):
    return round(float(v), 3)


def build_grid():
    """Exactly what plot_cloud_map.main() does for --all_stitches."""
    grid = {}
    for st in STITCHES:
        sd = os.path.join(ROOT, st)
        grid[st] = pcm.build_sm_panels(os.path.join(sd, f"refcrops_{st}.npz"), sd, None)
    return grid


def panel_stats(p):
    """COMs, mean D (= circle radii) and Delta, computed as plot_grid_sm does."""
    out = {}
    for sv in (pcm.OFF_S, pcm.ON_S):
        g = [f for f in p["fills"] if f["strength"] == sv]
        Y = np.array([f["y"] for f in g])
        out[sv] = {"com": Y.mean(0) @ p["B"], "r": float(np.mean([f["maha"] for f in g])), "n": len(g)}
    imp = 100.0 * (out[pcm.OFF_S]["r"] - out[pcm.ON_S]["r"]) / out[pcm.OFF_S]["r"]
    return out, imp


def tile_of(fname):
    return fname.split("__")[0]


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _pct(vals):
    """Percentile rank in [0, 1] of each value within its list."""
    order = np.argsort(np.argsort(vals))
    return order / max(len(vals) - 1, 1)


def select_fill_pairs(p, size):
    """5 (s=0, s=1) pairs per panel sharing tile + size + variant, spanning D,
    each from a different model variant."""
    fills =[f for f in p["fills"] if f["strength"] in (pcm.OFF_S, pcm.ON_S)]
    for sv in (pcm.OFF_S, pcm.ON_S):
        g = [f for f in fills if f["strength"] == sv]
        for f, q in zip(g, _pct(np.array([f["maha"] for f in g]))):
            f["_pct"] = float(q)
    pairs = defaultdict(dict)
    for f in fills:
        pairs[(f["model"], tile_of(f["file"]))][f["strength"]] = f
    def in_view(f):          # clickable: inside the figure's [-LIM, LIM] view (with a small inset)
        return float(np.max(np.abs(f["y"] @ p["B"]))) <= LIM - 0.5

    cands = [(k, v[pcm.OFF_S], v[pcm.ON_S]) for k, v in sorted(pairs.items())
             if pcm.OFF_S in v and pcm.ON_S in v and in_view(v[pcm.OFF_S]) and in_view(v[pcm.ON_S])]
    # score functions: lower is better
    targets = [
        ("far off",   lambda a, b: abs(a["_pct"] - 0.97)),
        ("far on",    lambda a, b: abs(b["_pct"] - 0.97)),
        ("both near", lambda a, b: abs(a["_pct"] - 0.03) + abs(b["_pct"] - 0.03)),
        ("median",    lambda a, b: abs(a["_pct"] - 0.5) + abs(b["_pct"] - 0.5)),
        ("LoRA helps", lambda a, b: -(a["_pct"] - b["_pct"])),
    ]
    used_v, used_t, chosen = set(), set(), []
    for why, fn in targets:
        pool = [c for c in cands if c[0][0] not in used_v and c[0][1] not in used_t
                and c not in chosen]
        best = min(pool, key=lambda c: (fn(c[1], c[2]), c[0]))
        chosen.append(best)
        used_v.add(best[0][0]); used_t.add(best[0][1])
    return [(why, c) for (why, _), c in zip(targets, chosen)]


def select_real(p, st, size, reprod, q, avoid_tiles):
    D = np.linalg.norm(p["y_real"], axis=1)
    rows = reprod[st][size]["rows"]
    assert len(rows) == len(D)
    target = np.quantile(D, q)
    order = np.argsort(np.abs(D - target))
    for i in order:
        if rows[i][0] not in avoid_tiles:
            return int(i), float(D[i]), rows[i]
    raise RuntimeError


def cmd_select(work):
    grid = build_grid()
    reprod = json.load(open(os.path.join(work, "reproduce.json")))
    jobs, sel = [], {"fills": {}, "real": {}}
    for st in STITCHES:
        real_q = {"128x128": 0.1, "256x256": 0.5, "512x512": 0.9}
        used_tiles = set()
        for size in pcm.SIZES:
            p = grid[st][size]
            px = size.split("x")[0]
            for why, ((variant, tile), a, b) in select_fill_pairs(p, size):
                for f, s in ((a, 0), (b, 1)):
                    key = f"{ABBR[st]}_{px}_{variant}_{tile}_s{s}"
                    stem = f["file"][:-4]
                    base = os.path.join(SWEEP, st, variant, size, stem)
                    jobs.append({"kind": "fill", "key": key, "png": base + ".png",
                                 "npy": base + ".npy", "mask": base + ".mask.png"})
                    sel["fills"][f"{st}|{size}|{variant}|{f['file']}"] = {
                        "key": key, "why": why,
                        "pair_file": (b if s == 0 else a)["file"]}
            i, D, (tile, y0, x0, side) = select_real(p, st, size, reprod, real_q[size], used_tiles)
            used_tiles.add(tile)
            key = f"{ABBR[st]}_{px}_real_{tile}_{y0}_{x0}"
            img_base = os.path.join(GB_REPO, "data_stitches", "holdouts-synthetic_damage", st, "images", tile)
            jobs.append({"kind": "real", "key": key, "tif": img_base + ".tif", "npy": img_base + ".npy",
                         "y0": y0, "x0": x0, "side": side})
            sel["real"][f"{st}|{size}"] = {"key": key, "i": i, "D": D, "tile": tile,
                                           "y0": y0, "x0": x0, "side": side, "q": real_q[size]}
    with open(os.path.join(work, "crop_jobs.json"), "w") as fh:
        json.dump(jobs, fh, indent=1)
    with open(os.path.join(work, "selection.json"), "w") as fh:
        json.dump(sel, fh, indent=1)
    print(f"{len(jobs)} crop jobs ({sum(j['kind'] == 'fill' for j in jobs)} fill, "
          f"{sum(j['kind'] == 'real' for j in jobs)} real)")


# ---------------------------------------------------------------------------
# final
# ---------------------------------------------------------------------------

def cmd_final(work):
    from PIL import Image
    grid = build_grid()
    sel = json.load(open(os.path.join(work, "selection.json")))
    meta = json.load(open(os.path.join(work, "crops", "crops_meta.json")))
    imgdir = os.path.join(OUT, "img")
    os.makedirs(imgdir, exist_ok=True)

    def webp(key):
        for kind, suf in (("rgb", ""), ("nrm", "_n")):
            src = os.path.join(work, "crops", f"{key}_{kind}.png")
            im = Image.open(src).convert("RGB")
            if kind == "rgb":
                im = im.resize((384, 384), Image.LANCZOS)   # whole tile; 384 px is sharp at card size
            im.save(os.path.join(imgdir, f"{key}{suf}.webp"),
                                                "WEBP", quality=80, method=6)
        return f"img/{key}.webp", f"img/{key}_n.webp"

    variants = sorted({f["model"] for st in STITCHES for p in grid[st].values() for f in p["fills"]})
    panels, n_sel_fill, n_sel_real = [], 0, 0
    for st in STITCHES:
        for size in pcm.SIZES:
            p = grid[st][size]
            B = p["B"]
            stats, imp = panel_stats(p)
            px = int(size.split("x")[0])
            fills_out, index_of = [], {}
            fl = [f for f in p["fills"] if f["strength"] in (pcm.OFF_S, pcm.ON_S)]
            for f in fl:
                xy = f["y"] @ B
                d = {"x": r3(xy[0]), "y": r3(xy[1]), "s": int(f["strength"]),
                     "D": r3(f["maha"]), "variant": f["model"], "id": tile_of(f["file"])}
                index_of[(f["model"], f["file"])] = len(fills_out)
                fills_out.append(d)
            for f in fl:
                s = sel["fills"].get(f"{st}|{size}|{f['model']}|{f['file']}")
                if not s:
                    continue
                d = fills_out[index_of[(f["model"], f["file"])]]
                img, nrm = webp(s["key"])
                m = meta[s["key"]]
                d.update({"img": img, "nrm": nrm, "bbox": m["bbox"], "file": f["file"],
                          "pair": index_of.get((f["model"], s["pair_file"]))})
                n_sel_fill += 1
            Pr = p["y_real"] @ B
            rs = sel["real"][f"{st}|{size}"]
            img, nrm = webp(rs["key"])
            real_sel = [{"i": rs["i"], "D": r3(rs["D"]), "id": rs["tile"],
                         "window_px": [rs["x0"], rs["y0"], rs["side"]],
                         "img": img, "nrm": nrm, "bbox": meta[rs["key"]]["bbox"]}]
            n_sel_real += 1
            ro, rn = stats[pcm.OFF_S]["r"], stats[pcm.ON_S]["r"]
            ticks = [-LIM, -LIM / 2, 0, LIM / 2, LIM]
            panels.append({
                "stitch": st, "stitch_label": pcm.STITCH_LABEL[st], "size": size, "hole_px": px,
                "title": f"{px} px hole", "xlabel": "PC1", "ylabel": f"{pcm.STITCH_LABEL[st]} PC2",
                "xlim": [-LIM, LIM], "ylim": [-LIM, LIM], "ticks": ticks,
                "tick_labels": ["", f"{-LIM/2:g}", "0", f"{LIM/2:g}", ""],
                "var_explained_pct": [round(float(v), 1) for v in p["var"]],
                "real_radius": r3(p["real_radius"]),
                "com_off": [r3(v) for v in stats[pcm.OFF_S]["com"]],
                "com_on": [r3(v) for v in stats[pcm.ON_S]["com"]],
                "mean_D_off": round(ro, 4), "mean_D_on": round(rn, 4), "delta_pct": round(imp, 2),
                "strip": {"off": f"{ro:.2f}", "on": f"{rn:.2f}", "delta": f"Δ{imp:+.0f}%"},
                "n_real": int(len(Pr)), "n_off": stats[pcm.OFF_S]["n"], "n_on": stats[pcm.ON_S]["n"],
                "real": [[r3(a), r3(b)] for a, b in Pr],
                "real_sel": real_sel,
                "fills": fills_out,
            })
    doc = {
        "figure": "Paper Figure 6 (fig:cloudmap), cloud_map_grid",
        "source": "plot_cloud_map.py --all_stitches --root _score_out_holdout_full --font_scale 2.2",
        "projection": ("per panel: standardise + whiten by the size-matched real reference covariance "
                       "(so Euclidean distance from the origin in 7-D = Mahalanobis D), then project onto "
                       "the top-2 right singular vectors of the pooled s=0/s=1 fills (not mean-centred)"),
        "features": pcm.FEATURES,
        "colours": {"off": pcm.OFF_C, "on": pcm.ON_C, "real": pcm.REAL_C},
        "variants": variants,
        "rows": STITCHES, "cols": pcm.SIZES,
        "panels": panels,
    }
    out_json = os.path.join(OUT, "points.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, separators=(",", ":"), ensure_ascii=False)
    print(f"wrote {out_json}: {os.path.getsize(out_json)/1e6:.2f} MB; "
          f"selectable fills {n_sel_fill}, real {n_sel_real}")
    verify_render(out_json, os.path.join(work, "verify_grid.png"))


def verify_render(json_path, out_png):
    """Re-draw the 3x3 grid from points.json alone, styled like the paper figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    doc = json.load(open(json_path, encoding="utf-8"))
    c = doc["colours"]
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 10.2), sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.05})
    for k, p in enumerate(doc["panels"]):
        ax = axes[k // 3][k % 3]
        R = np.array(p["real"])
        ax.scatter(R[:, 0], R[:, 1], s=8, c=c["real"], alpha=0.5, lw=0)
        ax.add_patch(Circle((0, 0), p["real_radius"], fill=False, ls="--", ec="#777"))
        ax.plot(0, 0, "k*", ms=13)
        for s, col, mk, key in ((0, c["off"], "o", "com_off"), (1, c["on"], "x", "com_on")):
            F = np.array([[d["x"], d["y"]] for d in p["fills"] if d["s"] == s])
            ax.scatter(F[:, 0], F[:, 1], s=22, c=col, marker=mk, alpha=0.6, lw=(1 if mk == "x" else 0))
            ax.add_patch(Circle(p[key], p["mean_D_off" if s == 0 else "mean_D_on"], fill=False, ec=col, lw=2))
            ax.plot(*p[key], marker="o" if s == 0 else "X", ms=11, color=col, mec="k")
        for d in p["fills"]:
            if "img" in d:
                ax.plot(d["x"], d["y"], "s", ms=9, mfc="none", mec="lime", mew=1.5, zorder=20)
        for r in p["real_sel"]:
            ax.plot(*p["real"][r["i"]], "D", ms=9, mfc="none", mec="magenta", mew=1.5, zorder=20)
        st = p["strip"]
        ax.text(0.03, 0.97, f"{st['off']}  {st['on']}  {st['delta']}", transform=ax.transAxes,
                va="top", fontsize=9, bbox=dict(fc="w", alpha=0.85))
        ax.set_xlim(*p["xlim"]); ax.set_ylim(*p["ylim"]); ax.set_box_aspect(1)
        ax.set_xticks(p["ticks"]); ax.set_xticklabels(p["tick_labels"])
        ax.set_yticks(p["ticks"]); ax.set_yticklabels(p["tick_labels"])
        ax.grid(ls=":", alpha=0.35)
        if k < 3:
            ax.set_title(p["title"])
        if k % 3 == 0:
            ax.set_ylabel(p["ylabel"])
        if k >= 6:
            ax.set_xlabel(p["xlabel"])
    fig.savefig(out_png, dpi=70, bbox_inches="tight")
    print("verify render ->", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "final"])
    ap.add_argument("--work", default=WORK_DEFAULT)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    {"select": cmd_select, "final": cmd_final}[a.stage](a.work)
