r"""GPU-box half of the interactive cloud-map build (read-only on the repo).

Runs on the gpubox with the ML venv, from a scratch folder (D:\_cloudmap_tmp),
with copies of distance_to_real.py, classical_surface_descriptors.py and
texture_fidelity.py from the repo placed in D:\_cloudmap_tmp\code (run python
with -B so nothing is written next to the repo code). It never writes inside
the repo or the sweep folder.

Sources (read only)
-------------------
  repo           D:\Projects\reintegrating_losses_paper
  real tiles     <repo>\data_stitches\holdouts-synthetic_damage\<stitch>\images\<stem>.tif  (1024^2 RGB uint8)
                 <repo>\data_stitches\holdouts-synthetic_damage\<stitch>\images\<stem>.npy  ((3,1024,1024) float32 normals)
  texture masks  <repo>\references\texture_masks\<stem>.png
  ref clouds     <repo>\_score_out_holdout_full\<stitch>\refcrops_<stitch>.npz
  fills          D:\reintegration_sweep_out\<stitch>\<variant>\<size>\<stem>.{png,npy,mask.png}

Modes
-----
  reproduce <out.json>
      Re-runs distance_to_real._crop_matrix (seed 0, 16 crops/tile, masked by
      references\texture_masks) for every stitch and loss size, but records the
      (tile, y0, x0) of every crop, recomputes its descriptor vector and compares
      it with the corresponding row of refcrops_<stitch>.npz.
      This mirrors the score_sm call in run_score_full_sweep.ps1:
        distance_to_real.py score_sm --real data_stitches\holdouts-synthetic_damage\<stitch>\images\*.npy
                                     --maskdir references\texture_masks --ref_npz ...

  crop <jobs.json> <outdir>
      For each job, writes <key>_rgb.png (the whole tile, 512x512) and
      <key>_nrm.png (normals of the fill region / real window only, 256x256),
      lossless (the local script converts to WebP), plus <outdir>\crops_meta.json
      with the region box in tile-normalised coords.
      Normals are visualised as rgb = (n + 1) / 2 * 255 (flat = lavender).
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "code"))

REPO = r"D:\Projects\reintegrating_losses_paper"
MASKDIR = os.path.join(REPO, "references", "texture_masks")
STITCHES = ["french_knot", "satin", "silk_purl"]
OUT_PX = 256           # normal-map image (the fill region / real window only)
FULL_PX = 512          # colour image (the whole 1024 px tile)


def real_paths(stitch):
    pat = os.path.join(REPO, "data_stitches", "holdouts-synthetic_damage", stitch, "images", "*.npy")
    return sorted(glob.glob(pat))


def reproduce(out_json):
    from pathlib import Path
    from classical_surface_descriptors import descriptors, feature_vector
    from distance_to_real import _texmask, _impute
    from texture_fidelity import load_normals

    res = {}
    for st in STITCHES:
        npz = np.load(os.path.join(REPO, "_score_out_holdout_full", st, f"refcrops_{st}.npz"))
        paths = real_paths(st)
        normals = {p: load_normals(Path(p)) for p in paths}
        res[st] = {}
        for key in [k for k in npz.files if k.startswith("X_")]:
            size = key[2:]
            side = int(npz[f"side_{size}"])
            k = 16
            rng = np.random.default_rng(0)            # _crop_matrix(seed=0), fresh per size
            rows, recs = [], []
            for p in paths:                           # verbatim control flow of _crop_matrix
                n = normals[p]
                H, W = n.shape[1:]
                mask = _texmask(MASKDIR, Path(p).stem, (H, W))
                if side >= H or side >= W:
                    if mask is None or mask.mean() > 0.999:
                        rows.append(feature_vector(descriptors(n)))
                        recs.append([Path(p).stem, 0, 0, int(min(H, W))])
                    continue
                got, tries, cap = 0, 0, k * 60
                while got < k and tries < cap:
                    tries += 1
                    y0 = int(rng.integers(0, H - side + 1))
                    x0 = int(rng.integers(0, W - side + 1))
                    if mask is not None and not mask[y0:y0 + side, x0:x0 + side].all():
                        continue
                    rows.append(feature_vector(descriptors(n[:, y0:y0 + side, x0:x0 + side])))
                    recs.append([Path(p).stem, y0, x0, side])
                    got += 1
            X = _impute(np.vstack(rows))
            Xn = npz[key]
            same_n = X.shape == Xn.shape
            diff = float(np.max(np.abs(X - Xn))) if same_n else None
            rel = float(np.max(np.abs(X - Xn) / (np.abs(Xn) + 1e-9))) if same_n else None
            res[st][size] = {"side": side, "n": len(recs), "n_npz": int(Xn.shape[0]),
                             "max_abs_diff": diff, "max_rel_diff": rel, "rows": recs}
            print(f"{st} {size}: n={len(recs)} npz={Xn.shape[0]} max|diff|={diff} max rel={rel}", flush=True)
    with open(out_json, "w") as fh:
        json.dump(res, fh)
    print("wrote", out_json)


def _nrm_rgb(n):
    """(3,H,W) float normals in [-1,1] -> (H,W,3) uint8, rgb = (n+1)/2*255."""
    return np.clip((np.transpose(n, (1, 2, 0)) + 1.0) * 0.5 * 255.0 + 0.5, 0, 255).astype(np.uint8)


def crop(jobs_json, outdir):
    import tifffile
    os.makedirs(outdir, exist_ok=True)
    jobs = json.load(open(jobs_json))
    meta = {}
    for j in jobs:
        if j["kind"] == "fill":
            rgb = np.asarray(Image.open(j["png"]).convert("RGB"))
            nrm = np.load(j["npy"]).astype(np.float32)
            if nrm.shape[0] != 3:
                nrm = np.transpose(nrm, (2, 0, 1))
            m = np.asarray(Image.open(j["mask"]).convert("L")) >= 128
            if m.shape != rgb.shape[:2]:
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.NEAREST)) >= 128
            ys, xs = np.where(m)
            ry0, ry1, rx0, rx1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            is_rect = bool(m.sum() == (ry1 - ry0) * (rx1 - rx0))
        else:
            rgb = tifffile.imread(j["tif"])
            if rgb.dtype != np.uint8:
                rgb = (rgb / rgb.max() * 255).astype(np.uint8)
            rgb = rgb[..., :3]
            nrm = np.load(j["npy"]).astype(np.float32)
            if nrm.shape[0] != 3:
                nrm = np.transpose(nrm, (2, 0, 1))
            ry0, rx0 = j["y0"], j["x0"]
            ry1, rx1 = ry0 + j["side"], rx0 + j["side"]
            is_rect = True
        H, W = rgb.shape[:2]
        assert nrm.shape[1:] == (H, W), (j["key"], nrm.shape, rgb.shape)
        # Colour: the whole tile, with bbox marking the fill region / real window on it.
        # Normals: only that region (what the descriptors are computed on), enlarged.
        c_rgb = Image.fromarray(rgb).resize((FULL_PX, FULL_PX), Image.LANCZOS)
        c_nrm = Image.fromarray(_nrm_rgb(nrm[:, ry0:ry1, rx0:rx1])).resize((OUT_PX, OUT_PX), Image.LANCZOS)
        c_rgb.save(os.path.join(outdir, f"{j['key']}_rgb.png"))
        c_nrm.save(os.path.join(outdir, f"{j['key']}_nrm.png"))
        reg = nrm[:, ry0:ry1, rx0:rx1]
        meta[j["key"]] = {
            "crop_box_px": [0, 0, W, H],                              # x0,y0,x1,y1 of the colour image on the tile
            "nrm_box_px": [rx0, ry0, rx1, ry1],                       # x0,y0,x1,y1 of the normal image on the tile
            "bbox": [round(rx0 / W, 4), round(ry0 / H, 4), round(rx1 / W, 4), round(ry1 / H, 4)],
            "region_is_rect": is_rect,
            "region_mean_normal": [round(float(v), 4) for v in reg.reshape(3, -1).mean(1)],
        }
        print(j["key"], meta[j["key"]]["crop_box_px"], flush=True)
    with open(os.path.join(outdir, "crops_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("wrote", len(meta), "crops")


if __name__ == "__main__":
    if sys.argv[1] == "reproduce":
        reproduce(sys.argv[2])
    elif sys.argv[1] == "crop":
        crop(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
