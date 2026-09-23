"""
Classical, training-free surface descriptors for embroidery normal maps.

Every feature here is a deterministic, citeable texture/geometry statistic — no
learned model sits on top of the (already learned) Marigold normal estimator.
The intent: characterise WHERE the real data of a given stitch type lives in
this feature space, then measure how far a generated fill sits from that point
(see distance_to_real.py). This generalises the original normal-variance idea to
a multivariate, interpretable feature vector.

Features (all computed on the normals or normal-derived fields):
  slope_mean_deg   mean zenith angle  (how tilted the surface is on average)
  slope_std_deg    spread of zenith   (knots: broad/steep sides; flat: narrow)
  roughness_rms    RMS deviation of nz from 1 (overall non-flatness)
  bump_density     convex local maxima per 1000 px  (french knots = bump field)
  bump_spacing_px  sqrt(area / bump_count)          (characteristic bump size)
  coherence        structure-tensor anisotropy (0 isotropic .. 1 oriented)
  pitch_px         radial-PSD dominant period (stitch spacing)

bump_density / bump_spacing / pitch are SCALE-DEPENDENT — only compare across
images at the same pixel scale (scale-match the reference to the fills).

Reuses loaders and the structure-tensor / PSD from texture_fidelity.py.

Usage
-----
# Per-tile feature CSV + per-group mean/std (group from parent dir by default):
venv\\Scripts\\python classical_surface_descriptors.py characterize ^
    --glob "references\\normals\\*\\*.npy" --out features_real.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from texture_fidelity import load_normals, _region, structure_tensor, stitch_pitch_px

EPS = 1e-8

# Order matters: this is the canonical feature vector used downstream.
FEATURES = ["slope_mean_deg", "slope_std_deg", "roughness_rms",
            "bump_density", "bump_spacing_px", "coherence", "pitch_px"]


def _bumps(normals: np.ndarray, reg: np.ndarray,
           min_dist: int = 7, k: float = 0.6) -> tuple[float, float]:
    """
    Convex-bump statistics from the surface curvature.

    Curvature proxy = divergence of the XY tilt field; a convex bump makes the
    normals splay outward, giving a local divergence maximum. Count smoothed
    local maxima above mean+k*std within the region (classical peak detection,
    no skimage), then report density and characteristic spacing.
    """
    nx, ny = normals[0], normals[1]
    div = np.gradient(nx, axis=1) + np.gradient(ny, axis=0)   # d nx/dx + d ny/dy
    div = gaussian_filter(div, 1.5)

    vals = div[reg]
    if vals.size < 16:
        return float("nan"), float("nan")
    thresh = vals.mean() + k * vals.std()

    peaks = (div == maximum_filter(div, size=min_dist)) & (div > thresh) & reg
    n = int(peaks.sum())
    area = int(reg.sum())
    density = n / (area / 1000.0 + EPS)                        # bumps per 1000 px
    spacing = float(np.sqrt(area / n)) if n > 0 else float("nan")
    return float(density), spacing


def descriptors(normals: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Full classical feature vector for one normal map (optionally masked)."""
    reg = _region(normals, mask)
    nz = np.clip(normals[2], -1.0, 1.0)
    zen = np.degrees(np.arccos(nz))                            # zenith angle (deg)

    z = zen[reg]
    slope_mean = float(z.mean()) if z.size else float("nan")
    slope_std = float(z.std()) if z.size else float("nan")
    roughness = float(np.sqrt(np.mean((1.0 - nz[reg]) ** 2))) if z.size else float("nan")

    density, spacing = _bumps(normals, reg)
    st = structure_tensor(normals, mask)
    pitch = stitch_pitch_px(normals, mask)

    return {
        "slope_mean_deg": slope_mean,
        "slope_std_deg": slope_std,
        "roughness_rms": roughness,
        "bump_density": density,
        "bump_spacing_px": spacing,
        "coherence": st["coherence"],
        "pitch_px": pitch["pitch_px"],
    }


def feature_vector(d: dict) -> np.ndarray:
    return np.array([d[f] for f in FEATURES], dtype=np.float64)


# ---------------------------------------------------------------------------
# CLI: characterize a set of normal maps
# ---------------------------------------------------------------------------

def _glob(pattern):
    import glob
    return sorted(glob.glob(pattern))


def cmd_characterize(args):
    paths = _glob(args.glob)
    print(f"Describing {len(paths)} normal maps …")
    rows = []
    for p in paths:
        p = Path(p)
        d = descriptors(load_normals(p))
        group = p.parent.name if args.group_from == "parent" else ""
        rows.append({"file": p.name, "group": group, **d})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "group", *FEATURES])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {args.out}")

    # per-group mean ± std for each feature
    from collections import defaultdict
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    print(f"\n{'group':<12}{'n':>4}  " + "".join(f"{f[:11]:>13}" for f in FEATURES))
    for g in sorted(by):
        rs = by[g]
        cells = []
        for f in FEATURES:
            v = np.array([r[f] for r in rs], dtype=np.float64)
            cells.append(f"{np.nanmean(v):>6.2f}±{np.nanstd(v):<5.2f}")
        print(f"{g:<12}{len(rs):>4}  " + "".join(f"{c:>13}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("characterize", help="Per-tile classical features -> CSV + group summary")
    c.add_argument("--glob", required=True)
    c.add_argument("--group_from", default="parent", choices=["parent", "none"])
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_characterize)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
