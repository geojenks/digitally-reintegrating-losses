"""
Re-crop the synthetic-damage holdout tiles at NATIVE 1024 from the source colour RTIs.

The 768px holdout tiles (data_stitches/holdouts-synthetic_damage/<stitch>/images,
named e{EMB}_r{RTI}_t{COL}_{ROW}) are pixel-exact 768 crops of the 7956x5312 source
colour RTIs (see Marigold training/tile_dataset.py). Filling them required upscaling
768 -> 1024 (Lanczos), which (a) forces the inpainter to work over blurred context and
(b) inflates the stitch pitch in pixels -- and the distance metric's pitch_px / bump_*
features are ABSOLUTE-PIXEL (classical_surface_descriptors.py, lines 20-21), so a fill
measured at 1024 is not on the same scale as a 768-native reference.

This pulls REAL 1024 pixels straight from the 42MP source instead: it decodes each
tile's grid origin (the exact tile_dataset convention) and "outsteps" 128px per side to
a 1024 window, clamped inward at the image edges (never upscaled). The 1024 window always
fully contains the original 768 tile, exposing ~128px more genuine context per side. This
is the same native re-crop done on 2026-06-04 for the silk_purl LoRA data.

Colour encoding matches the 768 tiles exactly: source uint16 -> uint8 via the high byte
(>> 8), the convention in tile_dataset.py. --verify re-extracts the 768 sub-window from
each fresh 1024 crop and asserts MSE == 0 against the existing holdout tile, so a wrong
origin / encoding is caught immediately.

Output (parallel tree, 768 layout preserved):
  data_stitches/holdouts-synthetic_damage_1024/<stitch>/images/<stem>.tif   (uint8 RGB)

NOTE on masks: this writes images only. Regenerate the inpaint masks at 1024 afterwards
by pointing generate_synthetic_masks.py at the _1024 root (it writes 128/256/512 centre-
square masks). The stale french_knot synthetic_masks do NOT carry over (its masks/ dir
holds only whole-motif masks that don't match tile stems), so all three stitches get
clean centre-square masks.

Run (either venv; needs numpy + tifffile):
  "Marigold training\\venv\\Scripts\\python" recrop_holdouts_1024.py --verify
  optional: --stitches satin silk_purl   --limit 5   --dry_run   --overwrite
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path(r"C:\Users\George (Stable Diff)\Documents\Projects\Reintegrating losses paper")
SRC_COLOUR   = Path(r"E:\embroidery_colour")
HOLDOUT_ROOT = PROJECT / "data_stitches" / "holdouts-synthetic_damage"
OUT_ROOT     = PROJECT / "data_stitches" / "holdouts-synthetic_damage_1024"

# Tile geometry -- identical to tile_dataset.py (do not change independently).
SRC_TILE = 768
OUT_TILE = 1024
PAD = (OUT_TILE - SRC_TILE) // 2          # 128px outstep per side
IMG_W, IMG_H = 7956, 5312

STEM_RE = re.compile(r"^e(\d+)_r(\d+)_t(\d+)_(\d+)$")
IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def tile_origins(span: int, tile: int) -> list[int]:
    """Tile origins along one axis with right/bottom reach-back (tile_dataset.py)."""
    origins = list(range(0, span - tile + 1, tile))
    last = span - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


COL_ORIGINS = tile_origins(IMG_W, SRC_TILE)   # 11 (idx 0..10)
ROW_ORIGINS = tile_origins(IMG_H, SRC_TILE)   # 7  (idx 0..6)


def colour_path(embr: int, rti: int) -> Path:
    name = f"P_RTI_{rti}.tif" if embr == 6 else f"RTI_{rti}.tif"
    return SRC_COLOUR / str(embr) / name


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def load_tile_u8(p: Path) -> np.ndarray:
    """Load an existing holdout tile as (H, W, 3) uint8 for verification."""
    a = tifffile.imread(p) if p.suffix.lower() in {".tif", ".tiff"} else \
        np.asarray(__import__("PIL.Image", fromlist=["Image"]).Image.open(p).convert("RGB"))
    if a.dtype != np.uint8:                    # paranoia: shouldn't happen for the 768 set
        a = (a >> 8).astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8)
    return a[..., :3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stitches", nargs="+", default=None,
                    help="subset of stitch dirs under the holdout root (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="cap tiles per stitch (0 = all)")
    ap.add_argument("--dry_run", action="store_true", help="print decode/window only, write nothing")
    ap.add_argument("--overwrite", action="store_true", help="rewrite existing _1024 tiles")
    ap.add_argument("--verify", action="store_true",
                    help="re-extract the 768 sub-window and assert MSE==0 vs the existing holdout tile")
    args = ap.parse_args()

    if not SRC_COLOUR.exists():
        sys.exit(f"Source colour root not reachable: {SRC_COLOUR} (is E: mounted?)")

    stitch_dirs = ([HOLDOUT_ROOT / s for s in args.stitches] if args.stitches
                   else sorted(p for p in HOLDOUT_ROOT.iterdir() if (p / "images").is_dir()))

    rti_cache: dict[Path, np.ndarray] = {}     # one RTI (~250MB uint16) held at a time
    n_ok = n_skip = n_edge = n_fail = 0
    max_mse = 0

    for sd in stitch_dirs:
        imgs = sorted(p for p in (sd / "images").iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if args.limit:
            imgs = imgs[:args.limit]
        out_dir = OUT_ROOT / sd.name / "images"
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{sd.name}] {len(imgs)} tiles -> {out_dir}")

        for img in imgs:
            m = STEM_RE.match(img.stem)
            if not m:
                print(f"  ! cannot parse {img.name}, skip"); n_fail += 1; continue
            embr, rti, ci, ri = (int(g) for g in m.groups())
            out_path = out_dir / f"{img.stem}.tif"
            if out_path.exists() and not args.overwrite and not args.dry_run and not args.verify:
                n_skip += 1; continue
            if ci >= len(COL_ORIGINS) or ri >= len(ROW_ORIGINS):
                print(f"  ! {img.stem}: col/row {ci},{ri} outside grid, skip"); n_fail += 1; continue

            x0, y0 = COL_ORIGINS[ci], ROW_ORIGINS[ri]          # exact 768 tile origin
            X0 = clamp(x0 - PAD, 0, IMG_W - OUT_TILE)
            Y0 = clamp(y0 - PAD, 0, IMG_H - OUT_TILE)
            edge = (X0 != x0 - PAD) or (Y0 != y0 - PAD)

            cp = colour_path(embr, rti)
            if not cp.exists():
                print(f"  ! missing source {cp}, skip"); n_fail += 1; continue
            if cp not in rti_cache:
                rti_cache.clear()
                arr = tifffile.imread(cp)
                if (arr.shape[1], arr.shape[0]) != (IMG_W, IMG_H):
                    print(f"  ! {cp.name} size {arr.shape[:2][::-1]} != {(IMG_W, IMG_H)}, skip")
                    n_fail += 1; continue
                rti_cache[cp] = arr
            arr = rti_cache[cp]

            crop8 = (arr[Y0:Y0 + OUT_TILE, X0:X0 + OUT_TILE] >> 8).astype(np.uint8)[..., :3]

            if args.verify:
                ox, oy = x0 - X0, y0 - Y0                       # where the 768 sits in the 1024
                sub = crop8[oy:oy + SRC_TILE, ox:ox + SRC_TILE]
                orig = load_tile_u8(img)
                if sub.shape == orig.shape:
                    mse = float(np.mean((sub.astype(np.int32) - orig.astype(np.int32)) ** 2))
                    max_mse = max(max_mse, mse)
                    if mse > 0:
                        print(f"  X {img.stem}: MSE={mse:.3f} (origin/encoding mismatch!)")
                        n_fail += 1
                else:
                    print(f"  ? {img.stem}: shape {sub.shape} vs {orig.shape}, cannot verify")

            if args.dry_run:
                print(f"  {img.stem}: src=({x0},{y0}) win=({X0},{Y0}){'  EDGE-SHIFT' if edge else ''}")
            else:
                tifffile.imwrite(out_path, crop8, compression="zlib")
            n_ok += 1
            n_edge += int(edge)

    print(f"\nDone: {n_ok} crops, {n_skip} skipped(existing), {n_edge} edge-shifted, {n_fail} failed.")
    if args.verify:
        print(f"Verify: max sub-window MSE = {max_mse:.4f}  ({'PIXEL-EXACT' if max_mse == 0 else 'MISMATCH — investigate'})")
    if not args.dry_run and n_ok and not args.verify:
        print("\nNext steps:")
        print("  1. Regenerate masks at 1024: run generate_synthetic_masks.py with BASE=holdouts-synthetic_damage_1024")
        print("  2. Point the sweep --images / --mask_root at the _1024 tree (drop --size 768 so FLUX fills native 1024)")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
