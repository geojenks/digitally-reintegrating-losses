"""
Distance-to-real-data evaluation in the classical descriptor space.

Core idea (the generalised variance metric): characterise WHERE real stitches of
a given type sit in the classical-descriptor space (centroid mu + spread Sigma),
then measure how far a generated fill sits from that real point, in units of the
real data's own spread — i.e. Mahalanobis distance. Report it both as named
per-property statements ("real bump_density = 0.42+/-0.08; LoRA 0.5 sigma away,
non-LoRA 4 sigma away") and as one overall distance, and plot it.

Modes
-----
validate : given several stitch types, check the descriptors actually place the
           types in distinct regions (per-tile nearest-centroid accuracy +
           confusion + a PCA scatter). Run this BEFORE trusting the metric.

score    : given a real reference set (one stitch type, scale-matched) and a set
           of generated fills (with sibling .mask.png), compute each fill's
           Mahalanobis distance to real + per-property z-scores, sorted by the
           LoRA strength parsed from the filename. Plots distance vs strength.

Usage
-----
venv\\Scripts\\python distance_to_real.py validate ^
    --type french_knot=references\\normals\\french_knot\\*.npy ^
    --type satin=references\\normals\\satin\\*.npy ^
    --type silk_purl=references\\normals\\silk_purl\\*.npy ^
    --plot validate_pca.png

venv\\Scripts\\python distance_to_real.py score ^
    --real "references\\normals\\french_knot\\*.npy" ^
    --fills "damage_restoration\\flux1_fill\\french_knot\\*\\*.npy" ^
    --strength_regex "_s([0-9.]+)" --out scores.csv --plot dist_vs_strength.png
"""

import argparse
import csv
import glob as _glob
import re
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from classical_surface_descriptors import descriptors, feature_vector, FEATURES
from texture_fidelity import load_normals, load_mask, _region

RIDGE = 1e-3   # covariance regularisation for stable inversion


# ---------------------------------------------------------------------------
# Feature-matrix helpers
# ---------------------------------------------------------------------------

def _texmask(maskdir, stem, hw):
    """Load a texture-validity mask <stem>.png (255=valid) for a tile, or None.

    Returned as a boolean array matched to (H, W); resized nearest if needed."""
    if not maskdir:
        return None
    mp = Path(maskdir) / f"{stem}.png"
    if not mp.exists():
        return None
    m = np.asarray(Image.open(mp).convert("L"))
    if m.shape != tuple(hw):
        m = np.asarray(Image.fromarray(m).resize((hw[1], hw[0]), Image.NEAREST))
    return m > 127


def _matrix(paths, masks=None, maskdir=None) -> np.ndarray:
    """Stack descriptor vectors for a list of normal-map paths -> (n, d).

    If maskdir is given, whole-tile descriptors are computed over the valid
    texture region only (the per-tile <stem>.png mask)."""
    rows = []
    for i, p in enumerate(paths):
        n = load_normals(Path(p))
        m = masks[i] if masks else None
        if m is None and maskdir:
            m = _texmask(maskdir, Path(p).stem, n.shape[1:])
        rows.append(feature_vector(descriptors(n, m)))
    return np.vstack(rows)


def _impute(X: np.ndarray) -> np.ndarray:
    """Replace NaNs with per-column median (some pitch/bump features can be NaN)."""
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med if np.isfinite(med) else 0.0
    return X


def _standardizer(X: np.ndarray):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return mu, sd, (X - mu) / sd


def _maha(x: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    d = x - mu
    return float(np.sqrt(max(d @ cov_inv @ d, 0.0)))


def _crop_matrix(paths, side: int, k: int, seed: int = 0, maskdir=None) -> np.ndarray:
    """Descriptor matrix over k random side x side crops of each tile.

    Scale-matching: descriptors like pitch_px / bump_* are pixel-scale-dependent,
    so a fill measured over an SxS mask region must be compared to REAL texture
    measured over the same SxS window — not over a whole tile. We random-crop the
    real tiles to `side` and describe each crop, building a window-matched cloud.

    If maskdir is given, only crops that fall ENTIRELY inside the tile's valid
    texture mask are kept (rejection sampling). Tiles whose valid region cannot
    fit `side` contribute fewer than k crops (or none) — cleaner cloud, less data.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in paths:
        n = load_normals(Path(p))
        H, W = n.shape[1:]
        mask = _texmask(maskdir, Path(p).stem, (H, W))
        if side >= H or side >= W:                 # tile smaller than window: use whole tile
            if mask is None or mask.mean() > 0.999:
                rows.append(feature_vector(descriptors(n)))
            continue
        got, tries, cap = 0, 0, k * 60
        while got < k and tries < cap:
            tries += 1
            y0 = int(rng.integers(0, H - side + 1))
            x0 = int(rng.integers(0, W - side + 1))
            if mask is not None and not mask[y0:y0 + side, x0:x0 + side].all():
                continue
            rows.append(feature_vector(descriptors(n[:, y0:y0 + side, x0:x0 + side])))
            got += 1
        if maskdir and got < k:
            print(f"    {Path(p).stem}: only {got}/{k} valid {side}px crops fit inside mask")
    if not rows:
        raise SystemExit(f"No valid {side}px crops inside any mask — masks too small for this window?")
    return _impute(np.vstack(rows))


def _ref_stats(X: np.ndarray):
    """Standardizer + regularised inverse covariance + centroid for a reference cloud."""
    X = _impute(X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    cov = np.cov(Z.T) + RIDGE * np.eye(len(FEATURES))
    return {"mu": mu, "sd": sd, "cov_inv": np.linalg.inv(cov),
            "centroid": Z.mean(0), "X": X, "std": X.std(0)}


def _savefig_both(path, dpi=130):
    """Save PNG plus a sibling SVG (vector copy for Inkscape touch-up in the paper)."""
    plt.savefig(path, dpi=dpi)
    plt.savefig(str(Path(path).with_suffix(".svg")))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def cmd_validate(args):
    types = {}
    for spec in args.type:
        name, pattern = spec.split("=", 1)
        types[name] = sorted(_glob.glob(pattern))
    if args.per_class:                                   # class-balance by random subsample
        rng = np.random.default_rng(args.ref_seed)
        for name in types:
            files = types[name]
            if len(files) > args.per_class:
                idx = rng.choice(len(files), args.per_class, replace=False)
                types[name] = [files[i] for i in sorted(idx)]
    for name in types:
        print(f"{name}: {len(types[name])} tiles")

    names = list(types)
    if args.window:
        print(f"\nWindowed separation check: {args.crops} random {args.window}x{args.window} "
              f"crops per tile (scale-matched to the fill mask region).")
        Xs = {t: _crop_matrix(types[t], args.window, args.crops, seed=args.ref_seed,
                              maskdir=args.maskdir) for t in names}
    else:
        Xs = {t: _impute(_matrix(types[t], maskdir=args.maskdir)) for t in names}
    Xall = np.vstack([Xs[t] for t in names])
    mu, sd, _ = _standardizer(Xall)
    Z = {t: (Xs[t] - mu) / sd for t in names}                  # standardized per type

    # pooled within-type covariance (LDA-style), regularised
    pooled = np.zeros((len(FEATURES), len(FEATURES)))
    for t in names:
        pooled += np.cov(Z[t].T) * (len(Z[t]) - 1)
    pooled /= (sum(len(Z[t]) for t in names) - len(names))
    pooled += RIDGE * np.eye(len(FEATURES))
    cov_inv = np.linalg.inv(pooled)
    centroids = {t: Z[t].mean(0) for t in names}

    # nearest-centroid classification by Mahalanobis distance
    conf = {t: {u: 0 for u in names} for t in names}
    correct = total = 0
    for t in names:
        for z in Z[t]:
            pred = min(names, key=lambda u: _maha(z, centroids[u], cov_inv))
            conf[t][pred] += 1
            correct += (pred == t)
            total += 1
    print(f"\nNearest-centroid accuracy (Mahalanobis): {correct}/{total} = {correct/total:.1%}")
    print("\nconfusion (row=true, col=pred):")
    print(f"{'':<12}" + "".join(f"{u:>12}" for u in names))
    for t in names:
        print(f"{t:<12}" + "".join(f"{conf[t][u]:>12}" for u in names))

    # inter-centroid Mahalanobis distances (how far apart the type clouds sit)
    print("\ninter-type centroid distance (Mahalanobis):")
    print(f"{'':<12}" + "".join(f"{u:>12}" for u in names))
    for t in names:
        print(f"{t:<12}" + "".join(f"{_maha(centroids[t], centroids[u], cov_inv):>12.2f}" for u in names))

    # PCA scatter (numpy SVD on standardized pooled features)
    Zpool = np.vstack([Z[t] for t in names])
    U, S, Vt = np.linalg.svd(Zpool - Zpool.mean(0), full_matrices=False)
    comp = Vt[:2].T
    plt.figure(figsize=(7, 6))
    for t in names:
        P = (Z[t] - Zpool.mean(0)) @ comp
        sc = plt.scatter(P[:, 0], P[:, 1], label=f"{t} (n={len(Z[t])})", alpha=0.7, s=22)
        c = (centroids[t] - Zpool.mean(0)) @ comp
        plt.scatter(*c, marker="X", s=200, color=sc.get_facecolor()[0],
                    edgecolor="k", linewidth=1.5, zorder=5)
    var = (S[:2] ** 2 / (S ** 2).sum()) * 100
    plt.xlabel(f"PC1 ({var[0]:.0f}%)"); plt.ylabel(f"PC2 ({var[1]:.0f}%)")
    wtxt = f"  (window {args.window}x{args.window})" if args.window else ""
    plt.title(f"Classical surface descriptors - stitch-type separation{wtxt}")
    plt.legend(); plt.tight_layout()
    _savefig_both(args.plot, dpi=130)
    print(f"\nPCA scatter -> {args.plot}  (+ svg)")


# ---------------------------------------------------------------------------
# score (fills vs real)
# ---------------------------------------------------------------------------

def _sibling_mask(p: Path, hw):
    mp = p.with_name(p.name[:-len(p.suffix)] + ".mask.png")
    return load_mask(mp, hw) if mp.exists() else None


def cmd_score(args):
    real_paths = sorted(_glob.glob(args.real))
    Xr = _impute(_matrix(real_paths, maskdir=args.maskdir))
    mu, sd = Xr.mean(0), Xr.std(0) + 1e-9
    Zr = (Xr - mu) / sd
    cov = np.cov(Zr.T) + RIDGE * np.eye(len(FEATURES))
    cov_inv = np.linalg.inv(cov)
    real_centroid = Zr.mean(0)

    print("real reference (mean +/- std):")
    for j, f in enumerate(FEATURES):
        print(f"  {f:<16} {mu[j]:.3f} +/- {Xr[:, j].std():.3f}")

    s_re = re.compile(args.strength_regex)
    size_re, cond_re = re.compile(r"__(\d+x\d+)__"), re.compile(r"__s[0-9.]+__([A-Za-z]+)")
    rows = []
    for p in sorted(_glob.glob(args.fills)):
        p = Path(p)
        m = s_re.search(p.stem)
        if not m:
            continue
        n = load_normals(p)
        mask = _sibling_mask(p, n.shape[1:])
        if _region(n, mask).sum() < 64:
            continue
        x = feature_vector(descriptors(n, mask))
        x = np.where(np.isfinite(x), x, mu)                    # impute NaN with real mean
        z = (x - mu) / sd
        row = {"file": p.name,
               "size": (size_re.search(p.stem) or [None, ""])[1] if size_re.search(p.stem) else "",
               "cond": (cond_re.search(p.stem).group(1) if cond_re.search(p.stem) else ""),
               "strength": float(m.group(1)),
               "maha_to_real": _maha(z, real_centroid, cov_inv)}
        for j, f in enumerate(FEATURES):                       # per-property z-scores
            row[f"z_{f}"] = (x[j] - mu[j]) / (Xr[:, j].std() + 1e-9)
        rows.append(row)

    rows.sort(key=lambda r: (r["size"], r["cond"], r["strength"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")

    # distance-to-real vs LoRA strength, one curve per hole size (+ all-sizes mean).
    # strength itself is the condition here: s0 = no-LoRA baseline, s>0 = LoRA.
    from collections import defaultdict
    by_size = defaultdict(list)
    for r in rows:
        by_size[(r["size"], r["strength"])].append(r["maha_to_real"])
    sizes = sorted({s for (s, _) in by_size})
    plt.figure(figsize=(7, 5))
    for sz in sizes:
        pts = sorted((st, np.mean(v)) for (s, st), v in by_size.items() if s == sz)
        if pts:
            xs, ys = zip(*pts)
            plt.plot(xs, ys, "o-", label=f"hole {sz}" if sz else "all")
    if len(sizes) > 1:                                         # overall trend across sizes
        allp = defaultdict(list)
        for r in rows:
            allp[r["strength"]].append(r["maha_to_real"])
        pts = sorted((st, np.mean(v)) for st, v in allp.items())
        xs, ys = zip(*pts)
        plt.plot(xs, ys, "k--", lw=2, label="all sizes")
    plt.xlabel("LoRA strength"); plt.ylabel("Mahalanobis distance to real (lower = closer)")
    plt.title("Distance from generated fill to real french-knot distribution")
    plt.legend(); plt.tight_layout(); _savefig_both(args.plot, dpi=130)
    print(f"distance-vs-strength plot -> {args.plot}  (+ svg)")


def cmd_score_sm(args):
    """Scale-matched scoring: each fill is compared to a real reference built from
    crops of the SAME pixel-window as the fill's mask region (removes the windowing
    bias in pitch_px / bump_* that inflates small-hole distances). Non-destructive —
    writes its own CSV/plot and dumps the per-size reference clouds for plotting."""
    real_paths = sorted(_glob.glob(args.real))
    if not real_paths:
        raise SystemExit(f"No real refs match {args.real}")

    s_re = re.compile(args.strength_regex)
    size_re, cond_re = re.compile(r"__(\d+x\d+)__"), re.compile(r"__s[0-9.]+__([A-Za-z]+)")

    # 1) gather fills with their raw descriptor vectors + actual pixel window side
    fills = []
    for p in sorted(_glob.glob(args.fills)):
        p = Path(p)
        m = s_re.search(p.stem)
        if not m:
            continue
        n = load_normals(p)
        mask = _sibling_mask(p, n.shape[1:])
        reg = _region(n, mask)
        if reg.sum() < 64:
            continue
        fills.append({
            "file": p.name,
            "size": (size_re.search(p.stem).group(1) if size_re.search(p.stem) else ""),
            "cond": (cond_re.search(p.stem).group(1) if cond_re.search(p.stem) else ""),
            "strength": float(m.group(1)),
            "x": feature_vector(descriptors(n, mask)),
            "side": int(round(np.sqrt(int(reg.sum())))),   # actual mask side in fill pixels
        })
    if not fills:
        raise SystemExit("No fills parsed.")

    # 2) per-size reference cloud (cached across variants — sides are deterministic)
    side_of = {}
    for f in fills:
        side_of.setdefault(f["size"], f["side"])
    cache = {}
    if args.ref_npz and Path(args.ref_npz).exists():
        z = np.load(args.ref_npz)
        cache = {k: z[k] for k in z.files}
    refs, rebuilt = {}, False
    for sz, side in sorted(side_of.items()):
        key = f"X_{sz}"
        if key in cache and int(cache.get(f"side_{sz}", -1)) == side:
            Xr_s = cache[key]
        else:
            print(f"  building real reference for {sz}: {args.crops} crops/tile @ {side}px ...")
            Xr_s = _crop_matrix(real_paths, side, args.crops, seed=args.ref_seed,
                                maskdir=args.maskdir)
            cache[key] = Xr_s
            cache[f"side_{sz}"] = np.array(side)
            rebuilt = True
        refs[sz] = _ref_stats(Xr_s)
    if args.ref_npz and rebuilt:
        np.savez(args.ref_npz, **cache)
        print(f"  cached references -> {args.ref_npz}")

    # 3) score each fill against its size-matched reference
    rows = []
    for f in fills:
        r = refs[f["size"]]
        x = np.where(np.isfinite(f["x"]), f["x"], r["mu"])
        z = (x - r["mu"]) / r["sd"]
        row = {"file": f["file"], "size": f["size"], "cond": f["cond"],
               "strength": f["strength"], "side": f["side"],
               "maha_to_real": _maha(z, r["centroid"], r["cov_inv"])}
        for j, ft in enumerate(FEATURES):
            row[f"z_{ft}"] = (x[j] - r["mu"][j]) / (r["std"][j] + 1e-9)
        rows.append(row)

    rows.sort(key=lambda r: (r["size"], r["cond"], r["strength"]))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")

    # per-size mean distance (the headline of the fix)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["size"], r["strength"])].append(r["maha_to_real"])
    print("\nscale-matched mean maha_to_real:")
    for sz in sorted({s for s, _ in agg}):
        cells = [f"s{st}={np.mean(agg[(sz, st)]):.2f}" for st in sorted({t for s, t in agg if s == sz})]
        print(f"  {sz} (side {side_of[sz]}px): " + "  ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Check the descriptors separate stitch types")
    v.add_argument("--type", action="append", required=True,
                   help="name=glob (repeatable)")
    v.add_argument("--plot", default="validate_pca.png")
    v.add_argument("--window", type=int, default=0,
                   help="if set, describe random WxW crops (scale-matched) instead of whole tiles")
    v.add_argument("--crops", type=int, default=16, help="random crops per tile when --window set")
    v.add_argument("--per_class", type=int, default=0,
                   help="cap each type to N random tiles (0=all) for class balance")
    v.add_argument("--maskdir", default=None,
                   help="dir of <stem>.png texture-validity masks; crops must fall fully inside")
    v.add_argument("--ref_seed", type=int, default=0)
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("score", help="Distance of fills to a real reference set")
    s.add_argument("--real", required=True, help="Glob of real normal maps (scale-matched)")
    s.add_argument("--fills", required=True, help="Glob of fill .npy (with sibling .mask.png)")
    s.add_argument("--strength_regex", default=r"_s([0-9.]+)")
    s.add_argument("--out", required=True)
    s.add_argument("--plot", default="dist_vs_strength.png")
    s.add_argument("--maskdir", default=None,
                   help="dir of <stem>.png texture-validity masks for the real reference")
    s.set_defaults(func=cmd_score)

    sm = sub.add_parser("score_sm", help="Scale-matched scoring (per-size window-matched reference)")
    sm.add_argument("--real", required=True, help="Glob of real normal maps to crop")
    sm.add_argument("--fills", required=True, help="Glob of fill .npy (with sibling .mask.png)")
    sm.add_argument("--strength_regex", default=r"_s([0-9.]+)")
    sm.add_argument("--out", required=True)
    sm.add_argument("--plot", default="dist_vs_strength_sm.png")
    sm.add_argument("--crops", type=int, default=16, help="random crops per real tile per size")
    sm.add_argument("--ref_seed", type=int, default=0)
    sm.add_argument("--ref_npz", default=None, help="cache/dump of per-size reference clouds")
    sm.add_argument("--maskdir", default=None,
                   help="dir of <stem>.png texture-validity masks; reference crops must fall fully inside")
    sm.set_defaults(func=cmd_score_sm)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
