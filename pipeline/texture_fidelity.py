"""
Arrangement-invariant fidelity metrics for reintegrated embroidery textures.

A generative loss-reintegration is NOT a reconstruction: a faithful fill places
realistic stitches in *different* positions than any original, so per-pixel /
aligned angular error is the wrong tool — it penalises exactly the behaviour we
want. These metrics instead ask whether a fill has the same *statistical and
structural character* as real stitches of its type, and whether it flows into
the surrounding original work. All metrics are translation/arrangement
invariant.

Consumes NORMAL MAPS (not RGB). Supported encodings, auto-detected by dtype:
  - 16-bit RGB TIFF  (GT)              : value/65535 * 2 - 1
  - 8-bit  RGB PNG/TIFF (viz strips)   : value/127.5    - 1
  - float .npy, channels-first or last : used as-is, assumed in [-1, 1]

Metric families implemented here (the dependency-light ones):
  (1) Structure tensor    -> dominant tilt orientation + coherence  ["thread direction"]
  (2) Radial power spectrum -> characteristic stitch pitch (px)     ["stitch scale"]
  (3) Sliced-Wasserstein over normal vectors + azimuth EMD          ["tilt distribution"]
  (6) Seam continuity     -> inner-fill vs outer-context agreement  ["seamlessness"]

Deferred to a second pass (need a CNN feature extractor): SIFID / KID / Gram
(family 4) and the Mahalanobis-to-real-cloud score (family 5).

Usage
-----
# Sanity check on real tiles BEFORE any reintegration output exists:
#   verify the metrics separate stitch types (satin = high coherence + clear
#   PSD peak; french knot = low coherence) and behave sensibly.
venv\\Scripts\\python texture_fidelity.py characterize ^
    --glob "tiles/normal/*.tif" ^
    --group_regex "^([a-z]+\\d+)_" ^
    --out characterize_real.csv

# Compare a fill against a reference (single ref map or a directory of refs),
# optionally with a fill mask for the seam-continuity metric:
venv\\Scripts\\python texture_fidelity.py compare ^
    --fill   fill_normal.tif ^
    --ref    "tiles/normal/*.tif" ^
    --mask   fill_mask.png ^
    --out    compare.csv

# LoRA-strength sweep: point at a folder of fills named with the strength,
# extract it via --strength_regex, score each against the same reference set.
venv\\Scripts\\python texture_fidelity.py sweep ^
    --glob   "damage_restoration\\flux1\\french_knot\\*.png" ^
    --ref    "tiles/normal/*.tif" ^
    --strength_regex "_s([0-9.]+)" ^
    --out    sweep_flux1_fk.csv
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import wasserstein_distance

EPS = 1e-8


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_normals(path: Path) -> np.ndarray:
    """Load a normal map as (3, H, W) float32 in [-1, 1], auto-detecting encoding."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.shape[0] == 3:                      # already (3, H, W)
            return arr
        return np.transpose(arr, (2, 0, 1))        # (H, W, 3) -> (3, H, W)

    if path.suffix.lower() in (".tif", ".tiff"):
        arr = tifffile.imread(path)
    else:
        arr = np.asarray(Image.open(path).convert("RGB"))

    if arr.dtype == np.uint16:
        normal = arr.astype(np.float32) / 65535.0 * 2.0 - 1.0
    elif arr.dtype == np.uint8:
        normal = arr.astype(np.float32) / 127.5 - 1.0
    else:                                          # float image, assume [-1, 1]
        normal = arr.astype(np.float32)
        if normal.max() > 1.5:                     # looks like [0, 255]
            normal = normal / 127.5 - 1.0

    return np.transpose(normal, (2, 0, 1))         # (3, H, W)


def load_mask(path: Path, hw: tuple[int, int]) -> np.ndarray:
    """Load an inpaint mask as bool (H, W); True = fill region (white in source)."""
    m = Image.open(path).convert("L").resize((hw[1], hw[0]), Image.NEAREST)
    return np.asarray(m) >= 128


def valid_mask(normals: np.ndarray) -> np.ndarray:
    """(H, W) bool: pixels with a non-degenerate normal vector."""
    return np.linalg.norm(normals, axis=0) > 1e-3


def _region(normals: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Combine validity with an optional region mask -> (H, W) bool."""
    v = valid_mask(normals)
    return v if mask is None else (v & mask)


# ---------------------------------------------------------------------------
# (1) Structure tensor  ->  dominant orientation + coherence
# ---------------------------------------------------------------------------

def structure_tensor(normals: np.ndarray, mask: np.ndarray | None = None,
                     sigma: float = 2.0) -> dict:
    """
    Orientation statistics of the XY tilt field (nx, ny). For small slopes the
    XY normal components behave like the gradient of the surface height, so the
    structure tensor of (nx, ny) recovers the dominant *tilt* direction and how
    strongly oriented the texture is.

    Returns
      tilt_orientation_deg : dominant tilt direction in [0, 180)
      thread_orientation_deg : tilt + 90 (ridges run perpendicular to tilt)
      coherence : 0 (isotropic, e.g. french knot) .. 1 (strongly oriented, e.g. satin)
    """
    nx, ny = normals[0], normals[1]
    reg = _region(normals, mask).astype(np.float32)

    # Smoothed second moments of the tilt field, masked.
    Jxx = gaussian_filter(nx * nx * reg, sigma)
    Jyy = gaussian_filter(ny * ny * reg, sigma)
    Jxy = gaussian_filter(nx * ny * reg, sigma)
    w   = gaussian_filter(reg, sigma) + EPS

    jxx, jyy, jxy = (Jxx / w)[reg > 0].mean(), (Jyy / w)[reg > 0].mean(), (Jxy / w)[reg > 0].mean()

    theta = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)          # radians, [-pi/2, pi/2]
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2) / (jxx + jyy + EPS)

    tilt_deg = np.degrees(theta) % 180.0
    return {
        "tilt_orientation_deg": float(tilt_deg),
        "thread_orientation_deg": float((tilt_deg + 90.0) % 180.0),
        "coherence": float(np.clip(coherence, 0.0, 1.0)),
    }


def orientation_diff_deg(a: float, b: float) -> float:
    """Smallest unsigned difference between two orientations on [0, 180)."""
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


# ---------------------------------------------------------------------------
# (2) Radial power spectrum  ->  characteristic stitch pitch
# ---------------------------------------------------------------------------

def stitch_pitch_px(normals: np.ndarray, mask: np.ndarray | None = None,
                    r_min: int = 4) -> dict:
    """
    Dominant spatial period of the surface, from the radially-averaged power
    spectrum of the tilt magnitude sqrt(nx^2 + ny^2). The peak frequency is the
    characteristic stitch spacing (period in pixels).

    A region mask is honoured by cropping to its bounding box (the FFT needs a
    rectangle); non-rectangular regions leak some energy, fine for relative use.
    """
    mag = np.sqrt(normals[0] ** 2 + normals[1] ** 2)
    if mask is not None:
        ys, xs = np.where(mask)
        if ys.size == 0:
            return {"pitch_px": float("nan"), "peak_freq": float("nan")}
        mag = mag[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    mag = mag - mag.mean()
    H, W = mag.shape
    N = min(H, W)
    mag = mag[:N, :N]                                       # square crop

    win = np.outer(np.hanning(N), np.hanning(N))            # reduce edge leakage
    P = np.abs(np.fft.fftshift(np.fft.fft2(mag * win))) ** 2

    cy, cx = N // 2, N // 2
    yy, xx = np.indices((N, N))
    r = np.round(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)).astype(int)
    radial = np.bincount(r.ravel(), P.ravel()) / (np.bincount(r.ravel()) + EPS)

    search = radial[r_min:N // 2]
    if search.size == 0 or not np.isfinite(search).any():
        return {"pitch_px": float("nan"), "peak_freq": float("nan")}
    r_peak = int(np.argmax(search)) + r_min
    return {"pitch_px": float(N / r_peak), "peak_freq": float(r_peak / N)}


# ---------------------------------------------------------------------------
# (3) Distribution distances over normal vectors
# ---------------------------------------------------------------------------

def _sample_normals(normals: np.ndarray, mask: np.ndarray | None,
                    n: int, rng: np.random.Generator) -> np.ndarray:
    """Flatten valid normals to (k, 3), subsampling to <= n for speed."""
    reg = _region(normals, mask)
    pts = normals[:, reg].T                                 # (k, 3)
    if pts.shape[0] > n:
        idx = rng.choice(pts.shape[0], n, replace=False)
        pts = pts[idx]
    return pts


def sliced_wasserstein(a: np.ndarray, b: np.ndarray,
                       n_proj: int = 128, seed: int = 0) -> float:
    """
    Sliced-Wasserstein distance between two clouds of unit normals (k, 3).
    Average 1-D Wasserstein over random spherical projections. Purely a
    distribution distance — ignores where each normal sat in the image.
    """
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_proj, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + EPS
    pa, pb = a @ dirs.T, b @ dirs.T                         # (ka, P), (kb, P)
    return float(np.mean([
        wasserstein_distance(pa[:, j], pb[:, j]) for j in range(n_proj)
    ]))


def azimuth_emd(a: np.ndarray, b: np.ndarray, bins: int = 36) -> float:
    """
    Earth-mover distance between tilt-azimuth histograms (interpretable companion
    to the SWD). Azimuth = atan2(ny, nx), weighted by tilt magnitude, folded to
    [0, pi) so opposite tilts of the same ridge coincide. Circular EMD.
    """
    def hist(p):
        az = (np.arctan2(p[:, 1], p[:, 0]) % np.pi)
        wt = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
        h, _ = np.histogram(az, bins=bins, range=(0, np.pi), weights=wt)
        return h / (h.sum() + EPS)
    ha, hb = hist(a), hist(b)
    # circular EMD on a ring: min over rotations of the cumulative-difference L1.
    centres = np.arange(bins)
    return float(min(
        wasserstein_distance(centres, centres, np.roll(ha, k), hb)
        for k in range(bins)
    ))


# ---------------------------------------------------------------------------
# (6) Seam continuity
# ---------------------------------------------------------------------------

def seam_continuity(normals: np.ndarray, fill_mask: np.ndarray,
                    band_px: int = 24, seed: int = 0) -> dict:
    """
    Agreement across the fill boundary: an inner band (inside the fill, near the
    edge) vs an outer band (intact context just outside). A seamless
    reintegration continues the surrounding thread direction, scale and tilt
    distribution.
    """
    from scipy.ndimage import binary_dilation, binary_erosion
    struct_iter = band_px
    inner = fill_mask & ~binary_erosion(fill_mask, iterations=struct_iter)
    outer = binary_dilation(fill_mask, iterations=struct_iter) & ~fill_mask

    si, so = structure_tensor(normals, inner), structure_tensor(normals, outer)
    rng = np.random.default_rng(seed)
    swd = sliced_wasserstein(
        _sample_normals(normals, inner, 20000, rng),
        _sample_normals(normals, outer, 20000, rng),
    )
    return {
        "seam_orientation_diff_deg": orientation_diff_deg(
            si["tilt_orientation_deg"], so["tilt_orientation_deg"]),
        "seam_coherence_diff": abs(si["coherence"] - so["coherence"]),
        "seam_swd": swd,
    }


# ---------------------------------------------------------------------------
# Per-tile feature vector + pairwise comparison
# ---------------------------------------------------------------------------

def characterize(normals: np.ndarray, mask: np.ndarray | None = None) -> dict:
    st = structure_tensor(normals, mask)
    pitch = stitch_pitch_px(normals, mask)
    return {**st, **pitch}


def compare(fill: np.ndarray, ref: np.ndarray,
            fill_mask: np.ndarray | None = None,
            ref_mask: np.ndarray | None = None,
            seed: int = 0) -> dict:
    """All arrangement-invariant fill-vs-reference metrics in one dict."""
    rng = np.random.default_rng(seed)
    fa = _sample_normals(fill, fill_mask, 20000, rng)
    rb = _sample_normals(ref, ref_mask, 20000, rng)

    sf, sr = structure_tensor(fill, fill_mask), structure_tensor(ref, ref_mask)
    pf, pr = stitch_pitch_px(fill, fill_mask), stitch_pitch_px(ref, ref_mask)
    return {
        "orientation_diff_deg": orientation_diff_deg(
            sf["tilt_orientation_deg"], sr["tilt_orientation_deg"]),
        "coherence_fill": sf["coherence"],
        "coherence_ref": sr["coherence"],
        "coherence_diff": abs(sf["coherence"] - sr["coherence"]),
        "pitch_fill_px": pf["pitch_px"],
        "pitch_ref_px": pr["pitch_px"],
        "pitch_ratio": pf["pitch_px"] / (pr["pitch_px"] + EPS),
        "swd": sliced_wasserstein(fa, rb, seed=seed),
        "azimuth_emd": azimuth_emd(fa, rb),
    }


def _load_ref_cloud(ref_glob: str, max_tiles: int, seed: int) -> np.ndarray:
    """Pool normals from many reference tiles into one cloud (k, 3)."""
    rng = np.random.default_rng(seed)
    paths = sorted(Path().glob(ref_glob)) or [Path(p) for p in _glob(ref_glob)]
    if len(paths) > max_tiles:
        paths = [paths[i] for i in rng.choice(len(paths), max_tiles, replace=False)]
    clouds = []
    for p in paths:
        n = load_normals(p)
        clouds.append(_sample_normals(n, None, 20000 // max(1, len(paths)) + 256, rng))
    return np.concatenate(clouds, axis=0)


def _glob(pattern: str) -> list[str]:
    """Glob that works with absolute Windows patterns (Path().glob can't)."""
    import glob
    return glob.glob(pattern)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], out: Path):
    if not rows:
        print("No rows to write.")
        return
    keys = list(rows[0].keys())
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {out}")


def cmd_characterize(args):
    paths = sorted(_glob(args.glob))
    print(f"Characterising {len(paths)} normal maps …")
    grp_re = re.compile(args.group_regex) if args.group_regex else None
    rows = []
    for p in paths:
        p = Path(p)
        feats = characterize(load_normals(p))
        group = (grp_re.search(p.stem).group(1) if grp_re and grp_re.search(p.stem)
                 else (p.parent.name if args.group_from == "parent" else ""))
        rows.append({"file": p.name, "group": group, **feats})
    _write_csv(rows, Path(args.out))
    _print_group_summary(rows)


def _print_group_summary(rows: list[dict]):
    """Mean coherence / pitch per group — the quick stitch-type separation check."""
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"\n{'group':<12} {'n':>4} {'coherence':>10} {'pitch_px':>10}")
    print("-" * 40)
    for g in sorted(by):
        rs = by[g]
        coh = np.nanmean([r["coherence"] for r in rs])
        pit = np.nanmean([r["pitch_px"] for r in rs])
        print(f"{g:<12} {len(rs):>4} {coh:>10.3f} {pit:>10.1f}")


def cmd_compare(args):
    fill = load_normals(Path(args.fill))
    fill_mask = load_mask(Path(args.mask), fill.shape[1:]) if args.mask else None

    ref_paths = sorted(_glob(args.ref))
    if len(ref_paths) == 1:
        ref = load_normals(Path(ref_paths[0]))
        row = compare(fill, ref, fill_mask=fill_mask)
    else:
        # Many references: pool into one cloud for the distribution metrics, and
        # average the scalar structure/pitch stats across the reference tiles.
        rng = np.random.default_rng(0)
        fa = _sample_normals(fill, fill_mask, 20000, rng)
        rb = _load_ref_cloud(args.ref, args.max_ref_tiles, seed=0)
        ref_feats = [characterize(load_normals(Path(p))) for p in ref_paths[:args.max_ref_tiles]]
        sf = structure_tensor(fill, fill_mask)
        pf = stitch_pitch_px(fill, fill_mask)
        ref_coh = np.nanmean([f["coherence"] for f in ref_feats])
        ref_pitch = np.nanmean([f["pitch_px"] for f in ref_feats])
        row = {
            "coherence_fill": sf["coherence"], "coherence_ref_mean": float(ref_coh),
            "coherence_diff": abs(sf["coherence"] - ref_coh),
            "pitch_fill_px": pf["pitch_px"], "pitch_ref_mean_px": float(ref_pitch),
            "pitch_ratio": pf["pitch_px"] / (ref_pitch + EPS),
            "swd": sliced_wasserstein(fa, rb),
            "azimuth_emd": azimuth_emd(fa, rb),
        }

    if fill_mask is not None:
        row.update(seam_continuity(fill, fill_mask))
    for k, v in row.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}")
    _write_csv([{"fill": Path(args.fill).name, **row}], Path(args.out))


def cmd_sweep(args):
    s_re = re.compile(args.strength_regex)
    rb = _load_ref_cloud(args.ref, args.max_ref_tiles, seed=0)
    ref_feats = [characterize(load_normals(Path(p)))
                 for p in sorted(_glob(args.ref))[:args.max_ref_tiles]]
    ref_coh = float(np.nanmean([f["coherence"] for f in ref_feats]))
    ref_pitch = float(np.nanmean([f["pitch_px"] for f in ref_feats]))

    size_re = re.compile(r"__(\d+x\d+)__")
    cond_re = re.compile(r"__s[0-9.]+__([A-Za-z]+)")
    piece_re = re.compile(r"^([^_]+)__")

    rows = []
    for p in sorted(_glob(args.glob)):
        p = Path(p)
        m = s_re.search(p.stem)
        if not m:
            print(f"  skip {p.name} (no strength match)")
            continue
        # restrict scoring to the fill region if a sibling mask exists, so the
        # surrounding (possibly different-stitch) context doesn't dilute the signal
        fill = load_normals(p)
        # sibling mask: <name>.mask.png (strip the .npy; can't use with_suffix —
        # the strength token, e.g. s1.0, contains a dot)
        mask_path = p.with_name(p.name[:-len(p.suffix)] + ".mask.png")
        fmask = load_mask(mask_path, fill.shape[1:]) if mask_path.exists() else None

        # some small masks (e.g. a 128px square that missed the item) are empty —
        # nothing was regenerated there, so there's no fill region to score
        if _region(fill, fmask).sum() < 64:
            print(f"  skip {p.name} (empty/tiny fill region)")
            continue

        rng = np.random.default_rng(0)
        fa = _sample_normals(fill, fmask, 20000, rng)
        sf, pf = structure_tensor(fill, fmask), stitch_pitch_px(fill, fmask)

        def grab(rx):
            mm = rx.search(p.stem)
            return mm.group(1) if mm else ""
        rows.append({
            "file": p.name,
            "piece": grab(piece_re), "size": grab(size_re), "cond": grab(cond_re),
            "strength": float(m.group(1)),
            "coherence": sf["coherence"], "coherence_diff": abs(sf["coherence"] - ref_coh),
            "pitch_px": pf["pitch_px"], "pitch_ratio": pf["pitch_px"] / (ref_pitch + EPS),
            "swd": sliced_wasserstein(fa, rb),
            "azimuth_emd": azimuth_emd(fa, rb),
        })
    rows.sort(key=lambda r: (r["size"], r["cond"], r["strength"]))
    _write_csv(rows, Path(args.out))

    # Aggregate by (size, cond, strength) so both the fairness baselines and the
    # per-strength trend are visible at a glance.
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        agg[(r["size"], r["cond"], r["strength"])].append(r)
    print(f"\n{'size':<10}{'cond':<7}{'str':>5}{'n':>4}{'swd':>9}{'azimuth':>9}{'coh_diff':>10}{'pitch_r':>9}")
    for key in sorted(agg):
        rs = agg[key]
        print(f"{key[0]:<10}{key[1]:<7}{key[2]:>5}{len(rs):>4}"
              f"{np.nanmean([r['swd'] for r in rs]):>9.4f}"
              f"{np.nanmean([r['azimuth_emd'] for r in rs]):>9.4f}"
              f"{np.nanmean([r['coherence_diff'] for r in rs]):>10.4f}"
              f"{np.nanmean([r['pitch_ratio'] for r in rs]):>9.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("characterize", help="Per-tile features over a glob -> CSV")
    c.add_argument("--glob", required=True)
    c.add_argument("--group_from", default="parent", choices=["parent", "none"])
    c.add_argument("--group_regex", default=None,
                   help="Regex with one capture group taken from the filename stem")
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_characterize)

    cmp = sub.add_parser("compare", help="One fill vs a reference (single map or glob)")
    cmp.add_argument("--fill", required=True)
    cmp.add_argument("--ref", required=True, help="Single normal map or a glob of refs")
    cmp.add_argument("--mask", default=None, help="Fill mask (enables seam metrics)")
    cmp.add_argument("--max_ref_tiles", type=int, default=40)
    cmp.add_argument("--out", required=True)
    cmp.set_defaults(func=cmd_compare)

    sw = sub.add_parser("sweep", help="LoRA-strength sweep of fills vs a reference set")
    sw.add_argument("--glob", required=True)
    sw.add_argument("--ref", required=True)
    sw.add_argument("--strength_regex", default=r"_s([0-9.]+)")
    sw.add_argument("--max_ref_tiles", type=int, default=40)
    sw.add_argument("--out", required=True)
    sw.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
