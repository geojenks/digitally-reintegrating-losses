"""
Run the fine-tuned Marigold-Normals model over the stitch-isolated RGB crops in
data_stitches/<type>/ to build per-stitch-type NORMAL reference sets.

Why: the fidelity metrics in texture_fidelity.py need a reference distribution of
*the same stitch type* as a given fill (a french-knot fill is judged against
french-knot normals, not whole-piece normals). The whole-piece GT tiles on E:
mix stitch types per embroidery, so they can't serve as a per-type reference.
The LoRA training crops in data_stitches/ are already stitch-isolated and
labelled by folder — this script turns their RGB into normals with the same
fine-tuned estimator used everywhere else in the pipeline.

Outputs, per source image:
  <out>/<type>/<stem>.npy   float32 (3, H, W) in [-1, 1]   (exact, for metrics)
  <out>/<type>/<stem>.png   8-bit normal viz               (for figures; --viz)

Model loading mirrors evaluate_high_variance_tiles.py: a baseline
MarigoldNormalsPipeline supplies VAE/scheduler/etc., and the fine-tuned UNet is
swapped in.

Usage
-----
cd "Marigold training"
venv\\Scripts\\python ..\\generate_type_normals.py ^
    --baseline  prs-eth/marigold-normals-v1-1 ^
    --finetuned models/marigold_normals_finetuned ^
    --src       ..\\data_stitches\\french_knot ..\\data_stitches\\satin ..\\data_stitches\\silk_purl ^
    --out       ..\\references\\normals ^
    --viz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Marigold-source lives next to evaluate_high_variance_tiles.py
sys.path.insert(0, str(Path(__file__).parent / "Marigold training" / "Marigold-source"))
from diffusers import UNet2DConditionModel          # noqa: E402
from marigold import MarigoldNormalsPipeline        # noqa: E402

DENOISING_STEPS = 4
PROCESSING_RES = 768
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def normal_to_u8(normal_float: np.ndarray) -> np.ndarray:
    """(3, H, W) float in [-1, 1] -> (H, W, 3) uint8, matching the eval encoding."""
    hwc = np.transpose(normal_float, (1, 2, 0))
    return ((hwc + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="Baseline MarigoldNormalsPipeline path/id (supplies VAE, scheduler …)")
    ap.add_argument("--finetuned", required=True,
                    help="Fine-tuned checkpoint dir containing a unet/ subfolder")
    ap.add_argument("--src", nargs="+", required=True,
                    help="One or more stitch-type crop directories")
    ap.add_argument("--out", default=None,
                    help="Output root; per-type subdirs are created. Not needed with --flat "
                         "(normals are written beside each source image).")
    ap.add_argument("--viz", action="store_true", help="Also write 8-bit normal PNGs")
    ap.add_argument("--limit", type=int, default=0, help="Cap images per type (0 = all)")
    ap.add_argument("--flat", action="store_true",
                    help="Write .npy next to each source image instead of an <out>/<type>/ subdir "
                         "(use for scoring reintegration fills in place)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute normals even if the target .npy already exists "
                         "(default: skip existing, so a big batch resumes after an interruption)")
    args = ap.parse_args()
    if not args.flat and not args.out:
        ap.error("--out is required unless --flat is given")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading baseline pipeline: {args.baseline}")
    pipe = MarigoldNormalsPipeline.from_pretrained(args.baseline).to(device)
    pipe.set_progress_bar_config(disable=True)

    ft_unet_path = Path(args.finetuned) / "unet"
    print(f"Loading fine-tuned UNet: {ft_unet_path}")
    pipe.unet = UNet2DConditionModel.from_pretrained(ft_unet_path).to(device)

    out_root = Path(args.out) if args.out else None
    for src_dir in args.src:
        src = Path(src_dir)
        type_name = src.name
        out_dir = src if args.flat else out_root / type_name
        out_dir.mkdir(parents=True, exist_ok=True)

        imgs = sorted(p for p in src.iterdir()
                      if p.is_file() and p.suffix.lower() in IMG_EXTS
                      and not p.stem.endswith(".mask"))      # skip inpaint masks
        if args.limit:
            imgs = imgs[:args.limit]
        print(f"\n[{type_name}] {len(imgs)} images -> {out_dir}")

        for i, img_path in enumerate(imgs, 1):
            out_npy = out_dir / f"{img_path.stem}.npy"
            if out_npy.exists() and not args.overwrite:        # resume: already done
                if i % 50 == 0 or i == len(imgs):
                    print(f"  {i}/{len(imgs)} (skip existing)")
                continue
            rgb = Image.open(img_path).convert("RGB")
            out = pipe(
                rgb,
                denoising_steps=DENOISING_STEPS,
                ensemble_size=1,
                processing_res=PROCESSING_RES,
                match_input_res=True,
                resample_method="bilinear",
                show_progress_bar=False,
            )
            normals = out.normals_np                          # (3, H, W) float [-1, 1]
            np.save(out_npy, normals.astype(np.float32))
            if args.viz and not args.flat:                    # avoid clobbering the RGB fill .png
                Image.fromarray(normal_to_u8(normals)).save(out_dir / f"{img_path.stem}.png")
            if i % 10 == 0 or i == len(imgs):
                print(f"  {i}/{len(imgs)}")

    if out_root is not None:                                  # subdir mode → per-type ref cloud
        print("\nDone. Next: validate per-type separation with\n"
              f"  python texture_fidelity.py characterize "
              f"--glob \"{out_root}\\*\\*.npy\" --group_from parent --out characterize_types.csv")
    else:
        print("\nDone (--flat): normals written beside each source image.")


if __name__ == "__main__":
    main()
