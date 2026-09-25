"""Build the "controllable structure" assets (procedural-init sweeps) for the project page.

Output: _public/docs/assets/structure/  (640x640 WebP, quality 76, manifest.json)

Sources (all 1024x1024, paths relative to the main repo root)
-------------------------------------------------------------
_demo_out/<run>_init.png   procedural init pasted into the stitch's mask(s)
_demo_out/<run>_final.png  generated fill from that init

All runs: FLUX.1-dev img2img through its inpainting pipeline (not FLUX.1-Fill)
plus the stitch LoRA, seed fixed within each stop, only the named parameter
changing between the runs of a stop. Denoise per stop as in STOPS below.

The frames are kept full frame (no crop), so every image of a stop is
framed identically. Dots in run names (thread_t4.5) become underscores in the
output file names.
"""
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "_demo_out"
OUT = ROOT / "_public" / "docs" / "assets" / "structure"
SIZE, QUALITY = 640, 76   # q80 came to 2.69 MB for the 32 images; q76 keeps it under 2.5 MB

# (run name, value label) per stop, in display order.
STOPS = [
    {"id": "knot", "stitch": "French knot", "variable": "knot radius", "unit": "px",
     "denoise": "0.65", "values": [("knot_r8", "8"), ("knot_r12", "12"), ("knot_r16", "16"), ("knot_r24", "24")]},
    {"id": "thread", "stitch": "Satin", "variable": "thread width", "unit": "px",
     "denoise": "0.5", "values": [("thread_t2", "2"), ("thread_t4.5", "4.5"), ("thread_t9", "9"), ("thread_t14", "14")]},
    # angle_step30 was run with --proc_satin_angle_step 30, but --brim 10 merges the satin
    # regions into one connected block, so every region got the base angle: it shows 0°.
    {"id": "angle", "stitch": "Satin", "variable": "thread angle", "unit": "°",
     "denoise": "0.5", "values": [("angle_step30", "0"), ("angle_uniform45", "45")]},
    {"id": "coil", "stitch": "Silk purl", "variable": "coil pitch", "unit": "px",
     "denoise": "0.8", "values": [("coil_c4", "4"), ("coil_c7", "7"), ("coil_c12", "12"), ("coil_c18", "18")]},
    {"id": "colour", "stitch": "All three stitches", "variable": "colour", "unit": "",
     "denoise": "0.65 / 0.5 / 0.8 (French knot / satin / silk purl)",
     "values": [("colour_A", "blue knots, gold satin, silver purl"),
                ("colour_B", "green knots, red satin, gold purl")]},
]


def convert(src, dst):
    im = Image.open(src).convert("RGB")
    assert im.size == (1024, 1024), (src, im.size)
    im.resize((SIZE, SIZE), Image.LANCZOS).save(dst, "WEBP", quality=QUALITY, method=6)
    return dst.stat().st_size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, total = {"size": SIZE, "stops": []}, 0
    for s in STOPS:
        vals = []
        for run, label in s["values"]:
            safe = run.replace(".", "_")
            files = {}
            for kind in ("init", "final"):
                files[kind] = f"{safe}_{kind}.webp"
                total += convert(SRC / f"{run}_{kind}.png", OUT / files[kind])
            vals.append({"label": label, "init": files["init"], "final": files["final"]})
        manifest["stops"].append({k: s[k] for k in ("id", "stitch", "variable", "unit", "denoise")} | {"values": vals})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    n = sum(len(s["values"]) for s in STOPS) * 2
    print(f"{n} images, {total / 1e6:.2f} MB -> {OUT}")


if __name__ == "__main__":
    main()
