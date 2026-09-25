"""Build the LoRA training-progression assets for the project page.

Output: _public/docs/assets/lora_training/  (320x320 WebP frames + manifest.json)

Sources
-------
* Training samples written by ai-toolkit on the GPU box:
    D:\\Projects\\reintegrating_losses_paper\\output\\<run>\\samples\\
  named ``{ms_timestamp}__{step:09d}_{promptIdx}.jpg`` (1024x1024, every 250
  steps). Sampling uses a fixed seed per prompt index (seed 42, walk_seed), so
  one prompt index across steps is the same composition evolving.
  Copy a local mirror first (only prompts 0/1 at 500-step multiples are used),
  e.g. per run:
      ssh gpubox "tar -cf - -C D:/Projects/reintegrating_losses_paper/output \
          <run>/samples/<file> ..." | tar -xf - -C <SRC>
  and pass --src <SRC> (a folder holding <run>/samples/*.jpg).
* Ground-truth tiles from the paper's Figure 3:
    _overleaf/figures/{french_knot,satin,silk_purl}.png  (main repo)
* Prompt text: _public/training/configs/<run>.yaml (sample.prompts[0:2]).
  No config survives for the FLUX.1-dev v1 runs; their step-0 samples are
  pixel-identical to the trigonly_v2 runs' step-0 samples for prompts 0 and 1,
  so the prompts (and seeds) are the same.

Where a step has two sample sets (satin FLUX v1 has two 7000-step sets, the
second from a later resume) the earliest timestamp is used. Missing steps are
simply omitted from the manifest.
"""
import argparse
import json
import re
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]          # main repo root
PUBLIC = ROOT / "_public"
OUT = PUBLIC / "docs" / "assets" / "lora_training"

STITCHES = ["french_knot", "satin", "silk_purl"]
RUN_PREFIX = {"french_knot": "french_knot_stitch", "satin": "satin_stitch",
              "silk_purl": "silk_purl"}
MODELS = [  # key, label, run suffix, prompt index used in paper Figure 3
    ("sdxl", "SDXL", "sdxl_lora_v1", 1),
    ("flux1_v1", "FLUX.1-dev (v1)", "flux1_lora_v1", 0),
    ("flux1_trigonly_v2", "FLUX.1-dev (released, trigonly_v2)", "flux1_lora_trigonly_v2", 0),
    ("qwen", "Qwen-Image", "qwen_lora_v1", 0),
    ("zimage", "Z-Image", "zimage_lora_v1", 0),
]
PROMPTS = (0, 1)
STEP_EVERY = 500
SIZE = 320
QUALITY = 80
PAT = re.compile(r"^(\d+)__(\d{9})_(\d+)\.jpg$")


def prompts_for(stitch, suffix):
    cfg_suffix = "flux1_lora_trigonly_v2" if suffix == "flux1_lora_v1" else suffix
    cfg = PUBLIC / "training" / "configs" / f"{RUN_PREFIX[stitch]}_{cfg_suffix}.yaml"
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    return c["config"]["process"][0]["sample"]["prompts"][:2], cfg.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="local mirror holding <run>/samples/*.jpg")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = {"models": [], "stitches": STITCHES, "gt": {}, "prompts": {},
                "figure3_prompt_index": {}, "frame_size": SIZE,
                "step_interval": STEP_EVERY}
    for key, label, suffix, fig_idx in MODELS:
        manifest["figure3_prompt_index"][key] = fig_idx
        m = {"key": key, "label": label, "figure3_prompt_index": fig_idx, "runs": {}}
        for stitch in STITCHES:
            run = f"{RUN_PREFIX[stitch]}_{suffix}"
            best = {}
            for f in (args.src / run / "samples").glob("*.jpg"):
                g = PAT.match(f.name)
                if not g:
                    continue
                ts, step, p = int(g[1]), int(g[2]), int(g[3])
                if p not in PROMPTS or step % STEP_EVERY:
                    continue
                if (step, p) not in best or ts < best[(step, p)][0]:
                    best[(step, p)] = (ts, f)
            steps = sorted({s for s, _ in best})
            # keep only steps present for both prompt indices
            steps = [s for s in steps if all((s, p) in best for p in PROMPTS)]
            files = {str(p): [] for p in PROMPTS}
            for s in steps:
                for p in PROMPTS:
                    name = f"{key}_{stitch}_p{p}_{s:05d}.webp"
                    im = Image.open(best[(s, p)][1]).convert("RGB")
                    im.resize((SIZE, SIZE), Image.LANCZOS).save(
                        OUT / name, "WEBP", quality=QUALITY, method=6)
                    files[str(p)].append(name)
            prompts, cfgname = prompts_for(stitch, suffix)
            m["runs"][stitch] = {"run": run, "steps": steps, "files": files,
                                 "prompts": {"0": prompts[0], "1": prompts[1]},
                                 "prompt_source": cfgname}
            manifest["prompts"].setdefault(stitch, {"0": prompts[0], "1": prompts[1]})
            print(f"{run}: {len(steps)} steps {steps}")
        manifest["models"].append(m)

    for stitch in STITCHES:
        name = f"gt_{stitch}.webp"
        Image.open(ROOT / "_overleaf" / "figures" / f"{stitch}.png").convert("RGB") \
            .resize((SIZE, SIZE), Image.LANCZOS).save(OUT / name, "WEBP",
                                                      quality=QUALITY, method=6)
        manifest["gt"][stitch] = name

    manifest["note"] = (
        "Frames are ai-toolkit training samples every 500 steps, fixed seed per "
        "prompt index; step 0 is the untrained base model. Steps that were never "
        "sampled are omitted, not interpolated: Qwen-Image satin has no samples "
        "for 4500-6500 (training resumed without sampling, only the final 7000 "
        "step was sampled), Qwen-Image silk purl has only steps 0 and 7000, and "
        "the released FLUX.1-dev trigonly_v2 runs stop at 5000. FLUX.1-dev v1 "
        "prompts are taken from the trigonly_v2 configs (the v1 configs were not "
        "kept; the step-0 samples of both runs are pixel-identical). Satin "
        "FLUX.1-dev v1 has two 7000-step sample sets; the first is used.")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
