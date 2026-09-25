r"""
staged_reintegrate.py — three-stage stitch reintegration from painted masks.

Given one image and the three masks from paint_stitch_masks.py
(<stem>__french_knot.png / __satin.png / __silk_purl.png), this fills each region
with its stitch in turn:  region 1 -> embfnchknt, region 2 -> embstn,
region 3 -> embslkprl. The model is loaded ONCE with all three stitch LoRAs as
named adapters; each stage just switches the active adapter + trigger. After each
stage the new fill is composited back ONLY inside that stage's mask, so earlier
stitches are preserved exactly (belt-and-suspenders on top of the inpaint
pipeline's own masking).

Models (matches run_sweep_allstitch.ps1 loading):
  sdxl       fast, mask-conditioned (good for iterating on masks/prompts/seeds)
  qwen_edit  best quality, instruction+mask, fp8 (the money shot)

Loads ONE model, runs --tries seeds, writes per-stage + final + a strip per try.

Usage
-----
# fast iteration
"Marigold training\\venv\\Scripts\\python" staged_reintegrate.py ^
    --image piece.tif --masks masks_out --model sdxl --tries 4 --out staged_out

# final quality
... --model qwen_edit --tries 3 --out staged_out

# a job bundle from the mask tool (see JOB_FORMAT.md): image, one mask per
# layer, per-layer colour/angle; --image/--masks not needed
... staged_reintegrate.py --job my_job.zip --model flux_base --lora_variant trigonly_v2
"""
from __future__ import annotations
import argparse
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # all checkpoints are local
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from reintegrate_holdouts import (quantize_transformer_fp8, QWEN_NEGATIVE,
                                  round16, png_meta)

# stitch -> (LoRA project name, trigger token)
STITCH = {
    "french_knot": ("french_knot_stitch", "embfnchknt"),
    "satin":       ("satin_stitch",       "embstn"),
    "silk_purl":   ("silk_purl",          "embslkprl"),
}

# per-stitch prompt template ({trig} = trained token). Descriptive/evocative to
# steer the fill against a different-stitch context; override per stitch from CLI.
PROMPT = {
    "french_knot": "{trig}, densely packed raised light red {trig} knots",
    "satin":       "{trig}, dense, smooth vertical light blue {trig}",
    "silk_purl":   "{trig}, metallic arches of dark brown {trig}, coiled wire",
}

# Qwen-Image-Edit is instruction-tuned -> imperative phrasing. The MASK already
# says where, so don't name the square; just say what to fill it with. Auto-used
# for qwen models; --prompt_* overrides still win.
QWEN_PROMPT = {
    "french_knot": "fill the masked area with densely packed raised light red {trig} texture",
    "satin":       "fill the masked area with dense smooth vertical light blue {trig} texture",
    "silk_purl":   "fill the masked area with metallic arches of dark brown {trig} texture, coiled wire",
}
# --job defaults: as above, colour from the layer ({colour}) and no fixed satin direction
# (the layer's angle sets it). "custom" = a job's own stitch with no prompt of its own.
JOB_PROMPT = {
    "french_knot": "{trig}, densely packed raised {colour} {trig} knots",
    "satin":       "{trig}, dense, smooth {colour} {trig}",
    "silk_purl":   "{trig}, metallic arches of {colour} {trig}, coiled wire",
    "custom":      "{trig}, dense {colour} {trig}",
}
JOB_QWEN_PROMPT = {
    "french_knot": "fill the masked area with densely packed raised {colour} {trig} texture",
    "satin":       "fill the masked area with dense smooth {colour} {trig} texture",
    "silk_purl":   "fill the masked area with metallic arches of {colour} {trig} texture, coiled wire",
    "custom":      "fill the masked area with {colour} {trig} texture",
}
# plain names for {colour}: nearest (RGB distance) wins
COLOUR_NAMES = {
    "black": (20, 20, 20), "dark grey": (70, 70, 70), "grey": (128, 128, 128),
    "light grey": (190, 190, 190), "white": (245, 245, 245), "cream": (238, 232, 214),
    "beige": (215, 195, 160), "light brown": (165, 120, 75), "brown": (115, 75, 40),
    "dark brown": (65, 42, 25), "gold": (205, 165, 60), "yellow": (235, 215, 70),
    "orange": (225, 130, 45), "light red": (215, 130, 130), "red": (190, 40, 40),
    "dark red": (115, 25, 30), "pink": (235, 165, 180), "purple": (115, 60, 135),
    "light blue": (150, 185, 220), "blue": (55, 90, 170), "dark blue": (25, 35, 90),
    "teal": (40, 125, 125), "light green": (150, 200, 130), "green": (60, 130, 60),
    "dark green": (30, 70, 35), "olive": (120, 120, 50),
}
# procedural default base colours (see proc_tex), for naming a layer with no colour
DEF_COL = {"french_knot": (200, 120, 120), "satin": (238, 232, 214),
           "silk_purl": (150, 110, 60), "flat": (128, 128, 128)}
LORA_FILE = {}                                          # --job custom stitch -> its LoRA file


def colour_name(rgb) -> str:
    return min(COLOUR_NAMES, key=lambda k: sum((a - b) ** 2 for a, b in zip(COLOUR_NAMES[k], rgb)))


def hex_rgb(h):
    """'#rrggbb' -> (r, g, b); None/'' -> None."""
    return tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) if h else None


def load_job(path):
    """Read a job bundle (.zip, folder, or its job.json; see JOB_FORMAT.md). Zips
    extract to a temp dir removed at exit. Paths resolve against the bundle root
    ('..' allowed, so sample jobs can reuse data/masks); returns job.json as a dict
    with absolute paths, plus _file (the job.json path)."""
    import atexit, json, re, shutil, tempfile, zipfile
    p = Path(path)
    if p.suffix.lower() == ".zip":
        root = Path(tempfile.mkdtemp(prefix="job_"))
        atexit.register(shutil.rmtree, root, True)
        with zipfile.ZipFile(p) as z:
            z.extractall(root)
        if not (root / "job.json").exists():            # zipped with a top folder
            subs = [d for d in root.iterdir() if (d / "job.json").exists()]
            root = subs[0] if len(subs) == 1 else root
        jf = root / "job.json"
    elif p.is_dir():
        root, jf = p, p / "job.json"
    else:
        root, jf = p.parent, p
    if not jf.exists():
        raise SystemExit(f"--job: no job.json in {path}")
    job = json.loads(jf.read_text(encoding="utf-8"))

    def bad(msg):
        return SystemExit(f"--job {path}: {msg}")

    def rp(s):
        return str((root / s).resolve())
    if job.get("format", "reintegration-job") != "reintegration-job":
        raise bad("format is not 'reintegration-job'")
    if int(job.get("version", 1)) > 1:
        raise bad(f"version {job['version']} is newer than this script understands (1)")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(job.get("name", ""))):
        raise bad("name must match [A-Za-z0-9_-]+")
    job["image"] = rp(job.get("image", "image.png"))
    if not Path(job["image"]).exists():
        raise bad(f"missing image: {job['image']}")
    stitches = job.get("stitches") or {}
    for k, s in stitches.items():
        if not re.fullmatch(r"[a-z0-9_]+", k) or not s.get("lora") or not s.get("trigger"):
            raise bad(f"stitch '{k}' needs a [a-z0-9_]+ key, a lora and a trigger")
        s["lora"] = rp(s["lora"])
        s.setdefault("init", k if k in DEF_COL else "flat")
        if s["init"] not in DEF_COL:
            raise bad(f"stitch '{k}': init must be one of {sorted(DEF_COL)}")
    layers = job.get("layers") or []
    if not layers:
        raise bad("no layers")
    isz, ids = Image.open(job["image"]).size, set()
    for L in layers:
        lid = str(L.get("id", ""))
        if not re.fullmatch(r"[a-z0-9_]+", lid) or lid in ids:
            raise bad(f"layer id '{lid}' must be unique and match [a-z0-9_]+")
        ids.add(lid)
        if L.get("stitch") not in STITCH and L.get("stitch") not in stitches:
            raise bad(f"layer '{lid}': unknown stitch '{L.get('stitch')}'")
        if L.get("colour") and not re.fullmatch(r"#[0-9a-fA-F]{6}", L["colour"]):
            raise bad(f"layer '{lid}': colour must be #rrggbb")
        L["mask"] = rp(L.get("mask", f"masks/{lid}.png"))
        if not Path(L["mask"]).exists():
            raise bad(f"layer '{lid}': missing mask {L['mask']}")
        if Image.open(L["mask"]).size != isz:
            print(f"WARN: layer '{lid}' mask is not the image size; resizing it")
    job["_file"] = str(jf)
    return job


def job_settings(ap, job):
    """job.json `settings` -> argparse defaults, so flags given on the command line
    still win. Keys must be real flags; the runner owns model/out."""
    acts = {a.dest: a for a in ap._actions}
    s = dict(job.get("settings") or {})
    if job.get("order") and "order" not in s:
        s["order"] = job["order"]
    out = {}
    for k, v in s.items():
        if k in ("model", "out", "image", "masks", "job"):
            print(f"WARN: job setting '{k}' ignored (the runner sets it)")
            continue
        if k not in acts or k == "help":
            raise SystemExit(f"--job: unknown setting '{k}' (not a staged_reintegrate.py flag)")
        if isinstance(v, list):
            v = ",".join(map(str, v))
        if isinstance(v, str) and acts[k].type is not None:
            v = acts[k].type(v)
        out[k] = v
    return out


# Local single-file checkpoints for the FLUX/Qwen variants; point these at your
# own downloads via environment variables. SDXL needs no local files (pulled
# from the Hugging Face hub on first run).
SF_QEDIT = os.environ.get("QWEN_EDIT_SAFETENSORS", r"models\qwen_image_edit_2511_bf16.safetensors")
QEDIT_COMPONENTS = "Qwen/Qwen-Image-Edit-2511"
SDXL_INPAINT_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"   # context-strong, LoRA-weak
SDXL_BASE_ID = "stabilityai/stable-diffusion-xl-base-1.0"             # masked img2img, LoRA-strong
SF_FLUX_DEV = os.environ.get("FLUX_DEV_SAFETENSORS", r"models\flux1-dev.safetensors")    # dev img2img, LoRA-strong
SF_FLUX_FILL = os.environ.get("FLUX_FILL_SAFETENSORS", r"models\flux1-fill-dev.safetensors")  # FLUX.1-Fill, context-strong
FLUX_COMPONENTS = "black-forest-labs/FLUX.1-dev"


def lora_path(stitch: str, model: str, variant: str = "v1", step: int = 0) -> str:
    if stitch in LORA_FILE:                              # --job custom stitch: its own file
        return LORA_FILE[stitch]
    lp, _ = STITCH[stitch]
    tok = "sdxl" if model.startswith("sdxl") else ("flux1" if model.startswith("flux") else "qwen")
    name = f"{lp}_{tok}_lora_{variant}"          # v1 | trigonly_v2 (flux1 only)
    fn = name if step <= 0 else f"{name}_{step:09d}"
    return f"output/{name}/{fn}.safetensors"


def paste_init(image: Image.Image, mask_L: Image.Image, tex_src, rng=None) -> Image.Image:
    """Crude starting point: tile a rectangle of real stitch texture (a training
    sample) into the mask and composite it in. Done PER CONNECTED REGION so each
    separate region gets the dense centre of the texture (a union bbox would let
    small regions land on sparse tile corners). With rng, the crop position is
    randomised (kept off the tile edges, where bare fabric tends to sit) for
    variety across runs. Meant to be followed by a fill at --denoise < 1 so the
    model refines rather than invents."""
    import cv2
    m8 = (np.asarray(mask_L) >= 128).astype(np.uint8)
    n, labels = cv2.connectedComponents(m8)
    if not callable(tex_src):
        tex_full = tex_src if isinstance(tex_src, Image.Image) else Image.open(tex_src).convert("RGB")
    result = image
    for reg_i, lab in enumerate(range(1, n)):
        if callable(tex_src):                            # per-region texture factory
            tex_full = tex_src(reg_i)
        comp = labels == lab
        ys, xs = np.nonzero(comp)
        l, u, r, lo = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        w, h = r - l, lo - u
        tex = tex_full
        if tex.width > w or tex.height > h:              # crop big training tiles
            cw, ch = min(tex.width, w), min(tex.height, h)
            if rng is not None:                          # random crop, inset 1/8 from edges
                mx, my = (tex.width - cw), (tex.height - ch)
                ix, iy = min(tex.width // 8, mx // 2), min(tex.height // 8, my // 2)
                x0 = rng.randint(ix, mx - ix) if mx > 2 * ix else mx // 2
                y0 = rng.randint(iy, my - iy) if my > 2 * iy else my // 2
            else:                                        # centre crop
                x0, y0 = (tex.width - cw) // 2, (tex.height - ch) // 2
            tex = tex.crop((x0, y0, x0 + cw, y0 + ch))
        tile = Image.new("RGB", (int(w), int(h)))
        for yy in range(0, h, tex.height):
            for xx in range(0, w, tex.width):
                tile.paste(tex, (xx, yy))
        patched = result.copy()
        patched.paste(tile, (int(l), int(u)))
        comp_L = Image.fromarray((comp * 255).astype(np.uint8), "L")
        result = Image.composite(patched, result, comp_L)
    return result


def proc_tex(stitch: str, rng, size: int = 512, knot_r: int = 0,
             satin_thread: float = 0.0, coil: float = 0.0,
             noise: float = 0.0, col=None, angle: float = 0.0) -> Image.Image:
    """Procedural stand-in texture: right bumpiness and periodicity, no realism —
    the denoise pass supplies that. Rough source-matched colours (see PROMPT).
    knot_r / satin_thread / coil override the period parameters (0 = randomised),
    for controllable varieties (knot size, thread spacing, coil pitch)."""
    a = np.zeros((size, size, 3), np.float32)
    noise_sigma = 8                                     # enough grit that denoise invents structure
    if stitch == "french_knot":
        base = np.array(col or rng.choice([(200, 120, 120), (215, 150, 150), (225, 190, 180)]),
                        np.float32)
        a[:] = base * 0.55                              # shadowed ground between knots
        r = knot_r or rng.randint(11, 15)               # knot radius (working-image px)
        step = int(r * 1.9)
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        bump = np.clip(1.15 - (np.sqrt(yy ** 2 + xx ** 2) / r) ** 1.5, 0, 1)
        for row, y in enumerate(range(r, size - r, step)):
            for x in range(r + ((step // 2) if row % 2 else 0), size - r, step):
                cy, cx = y + rng.randint(-3, 3), x + rng.randint(-3, 3)
                if not (r <= cy < size - r and r <= cx < size - r):
                    continue
                col = np.clip(base * (0.75 + 0.5 * rng.random()), 0, 255)
                sl = a[cy - r:cy + r + 1, cx - r:cx + r + 1]
                sl[:] = sl * (1 - bump[..., None] * 0.95) + col * bump[..., None] * 0.95
    elif stitch == "satin":
        base = np.array(col or (238, 232, 214), np.float32)    # cream
        tp_ = satin_thread or 4.5                       # thread width (working-image px)
        th = np.deg2rad(angle)                          # 0 = vertical threads
        yy, xx = np.mgrid[0:size, 0:size]
        phase = xx * np.cos(th) + yy * np.sin(th)
        thread = 0.82 + 0.18 * np.abs(np.sin(np.pi * phase / tp_))
        a[:] = base * thread[..., None]                 # no sheen banding: it survived low
                                                        # denoise as grey wall stripes
    elif stitch == "flat":                              # --job custom stitch, no structure
        a[:] = np.array(col or DEF_COL["flat"], np.float32)
    else:                                               # silk_purl: coiled-wire rows
        base = np.array(col or (150, 110, 60), np.float32)     # brown-gold
        cp = coil or rng.uniform(6.0, 8.0)              # coil pitch (working-image px)
        y = np.arange(size)
        coil_w = 0.5 + 0.5 * np.abs(np.sin(np.pi * y / cp))             # deep coil grooves
        band = 0.8 + 0.2 * np.sin(2 * np.pi * y / (4 * cp))             # row shading
        shim = 0.85 + 0.15 * np.sin(2 * np.pi * np.arange(size) / rng.randint(40, 90))
        a[:] = base * (coil_w * band)[:, None, None] * shim[None, :, None]
        noise_sigma = 10                                # purl needs more grit to denoise into wire
    a += np.random.default_rng(rng.randint(0, 10 ** 9)).normal(0, noise or noise_sigma, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def keep_dark(pasted: Image.Image, orig: Image.Image, mask_L: Image.Image,
              thresh: float) -> Image.Image:
    """Carry the original's dark drawn features (design underdrawing, window
    patches) through the paste-init: inside the mask, pixels darker than
    `thresh` are blended back over the pasted texture so the denoise pass
    renders them as stitched features instead of erasing them."""
    from PIL import ImageFilter
    o = np.asarray(orig).astype(np.float32)
    p = np.asarray(pasted).astype(np.float32)
    lum = o.mean(2)
    alpha = np.clip((thresh - lum) / 40.0, 0, 1)
    alpha[np.asarray(mask_L) < 128] = 0
    alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8), "L")
                       .filter(ImageFilter.GaussianBlur(1))).astype(np.float32) / 255.0
    out = p * (1 - alpha[..., None]) + o * alpha[..., None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


def feather(mask_L: Image.Image, px: int) -> Image.Image:
    """Soften a binary composite mask: grow 1px then Gaussian-blur, so the fill
    blends over a ~px band instead of a hard cut-out edge."""
    from PIL import ImageFilter
    return mask_L.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(px))


def dilate(mask_L: Image.Image, px: int) -> Image.Image:
    """Dilate a binary mask by px (no-op for px<=0)."""
    from PIL import ImageFilter
    return mask_L.filter(ImageFilter.MaxFilter(2 * px + 1)) if px > 0 else mask_L


def grow_composite(mask_L, others_L, grow, feath, sib=0):
    """Composite-time mask: dilate by `grow` px so the model's own blended edge
    (which extends a little past the mask) is kept in the picture, while
    subtracting the other stitches' regions so the grown band never eats a
    sibling fill; then optionally feather. Unlike --mask_grow this changes only
    what is pasted back, never what the model is asked to fill."""
    from PIL import ImageFilter
    cm = mask_L
    if grow > 0:
        cm = cm.filter(ImageFilter.MaxFilter(2 * grow + 1))
        if others_L is not None:
            a = np.asarray(cm).copy()
            a[np.asarray(others_L) >= 128] = 0
            cm = Image.fromarray(np.maximum(a, np.asarray(mask_L)), "L")
    if feath > 0:
        cm = feather(cm, feath)
    if sib > 0 and others_L is not None:
        # soften ONLY the stitch-vs-stitch boundary: alpha-blend the composite mask
        # in a band around the sibling regions, keeping the outer edge hard
        near = np.asarray(dilate(others_L, sib + 2)) >= 128
        from PIL import ImageFilter
        blur = np.asarray(cm.filter(ImageFilter.GaussianBlur(sib)))
        a = np.asarray(cm).copy()
        a[near] = blur[near]
        cm = Image.fromarray(a, "L")
    return cm


def rim_mask(mask_L, px):
    """A band straddling the region boundary (~px inside + px outside), for a
    seam-blending inpaint pass after the body has been filled by base img2img."""
    import cv2
    m8 = (np.asarray(mask_L) >= 128).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    dil = cv2.dilate(m8, k)
    ero = cv2.erode(m8, k)
    return Image.fromarray((((dil > 0) & (ero == 0)).astype(np.uint8) * 255), "L")


# ---------------------------------------------------------------------------
# loaders (one model, three LoRA adapters)
# ---------------------------------------------------------------------------
def build_sdxl(loras, base_id):
    from diffusers import StableDiffusionXLInpaintPipeline
    print(f"Loading {base_id} …")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(base_id, torch_dtype=torch.float16)
    for name, p in loras.items():
        print(f"  + LoRA {name}: {p}")
        pipe.load_lora_weights(os.path.dirname(p), weight_name=os.path.basename(p), adapter_name=name)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe, dict(size=1024, steps=40, guidance=7.5)


def build_flux(loras, transformer_file=SF_FLUX_DEV, components_id=FLUX_COMPONENTS):
    """FLUX.1-dev img2img (LoRA-strong analog of sdxl_base). Assembled from the
    local single-file dev transformer + cached FLUX.1-dev components; bf16 + offload."""
    from diffusers import FluxInpaintPipeline, FluxTransformer2DModel
    print(f"Loading FLUX.1-dev transformer (single file): {transformer_file}")
    transformer = FluxTransformer2DModel.from_single_file(transformer_file, torch_dtype=torch.bfloat16)
    print(f"Assembling FluxInpaintPipeline from cached components: {components_id}")
    pipe = FluxInpaintPipeline.from_pretrained(components_id, transformer=transformer, torch_dtype=torch.bfloat16)
    for name, p in loras.items():
        print(f"  + LoRA {name}: {p}")
        pipe.load_lora_weights(os.path.dirname(p), weight_name=os.path.basename(p), adapter_name=name)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe, dict(size=1024, steps=28, guidance=3.5)


def build_flux_fill(loras, transformer_file=SF_FLUX_FILL, components_id=FLUX_COMPONENTS):
    """FLUX.1-Fill (context-strong inpaint). Assembled from the local single-file
    Fill transformer + cached FLUX.1-dev components (VAE/T5/CLIP shared)."""
    from diffusers import FluxFillPipeline, FluxTransformer2DModel
    print(f"Loading FLUX.1-Fill transformer (single file): {transformer_file}")
    transformer = FluxTransformer2DModel.from_single_file(transformer_file, torch_dtype=torch.bfloat16)
    print(f"Assembling FluxFillPipeline from cached components: {components_id}")
    pipe = FluxFillPipeline.from_pretrained(components_id, transformer=transformer, torch_dtype=torch.bfloat16)
    for name, p in loras.items():
        print(f"  + LoRA {name}: {p}")
        pipe.load_lora_weights(os.path.dirname(p), weight_name=os.path.basename(p), adapter_name=name)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe, dict(size=1024, steps=30, guidance=30.0)


def build_qwen_edit(loras, transformer_file, fp8=True, quant_cache=None):
    from diffusers import QwenImageEditInpaintPipeline, QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration
    print("Borrowing text_encoder from Qwen/Qwen-Image")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image", subfolder="text_encoder", torch_dtype=torch.bfloat16)

    # quant_cache: a torch.save of the fully-assembled transformer (fp8 base + the
    # 3 bf16 stitch adapters baked in). Reloading it skips the 23GB single-file load,
    # the LoRA injection, AND the ~10-min fp8 quantise. The cache is tied to THIS set
    # of LoRAs/order; delete it if those change.
    if quant_cache and Path(quant_cache).exists():
        print(f"Loading cached quantised transformer: {quant_cache}")
        transformer = torch.load(quant_cache, weights_only=False)
        pipe = QwenImageEditInpaintPipeline.from_pretrained(
            QEDIT_COMPONENTS, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
    else:
        print(f"Loading Qwen-Edit transformer (single file): {transformer_file}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            transformer_file, config=QEDIT_COMPONENTS, subfolder="transformer", torch_dtype=torch.bfloat16)
        pipe = QwenImageEditInpaintPipeline.from_pretrained(
            QEDIT_COMPONENTS, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch.bfloat16)
        for name, p in loras.items():                 # ALL adapters before fp8 quantise
            print(f"  + LoRA {name}: {p}")
            pipe.load_lora_weights(os.path.dirname(p), weight_name=os.path.basename(p), adapter_name=name)
        if fp8:
            quantize_transformer_fp8(pipe.transformer)
        if quant_cache:                                # save AFTER quantise, BEFORE offload (CPU tensors)
            print(f"Saving quantised transformer -> {quant_cache} (~20GB, one-off)")
            torch.save(pipe.transformer, quant_cache)
    print(f"adapters present: {pipe.get_list_adapters()}")   # verify the 3 stitch LoRAs survived
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    return pipe, dict(size=1024, steps=30, guidance=4.0)


# ---------------------------------------------------------------------------
# one stage
# ---------------------------------------------------------------------------
def fill_stage(pipe, kind, image, mask_L, prompt, size, steps, guidance, seed, denoise=1.0):
    gen = torch.Generator("cpu").manual_seed(seed)
    if kind == "qwen_edit":
        # blank the masked region to neutral grey in the reference, else the Edit
        # model copies the existing pixels back (same trick as QwenEditInpaint.fill)
        a = np.asarray(image).copy()
        m = np.asarray(mask_L.resize(image.size, Image.NEAREST)) >= 128
        a[m] = 128
        ref = Image.fromarray(a)
        return pipe(image=ref, mask_image=mask_L, prompt=prompt, negative_prompt=QWEN_NEGATIVE,
                    height=size, width=size, num_inference_steps=steps,
                    true_cfg_scale=guidance, guidance_scale=1.0, strength=1.0,
                    max_sequence_length=512, generator=gen).images[0]
    if kind == "flux_fill":                              # FluxFillPipeline has no strength knob
        return pipe(prompt=prompt, image=image, mask_image=mask_L, height=size, width=size,
                    num_inference_steps=steps, guidance_scale=guidance,
                    max_sequence_length=512, generator=gen).images[0]
    if kind.startswith("flux"):                          # flux_base = FluxInpaint img2img
        return pipe(prompt=prompt, image=image, mask_image=mask_L, height=size, width=size,
                    num_inference_steps=steps, guidance_scale=guidance, strength=denoise,
                    max_sequence_length=512, generator=gen).images[0]
    return pipe(prompt=prompt, image=image, mask_image=mask_L, height=size, width=size,
                num_inference_steps=steps, guidance_scale=guidance, strength=denoise,
                generator=gen).images[0]


def dark_stitch_pass(pipe, args, running, stamp_alpha, size, steps, guidance,
                     seed, denoise, strength):
    """Re-render the darkened design lines/windows as stitched thread: satin LoRA,
    low denoise, masked to the dilated dark features only. Expects `running` to
    already carry the --stamp_dark darkening (the pass refines dark marks into
    dark stitching; it can't invent darkness that isn't there)."""
    dm = Image.fromarray(((stamp_alpha > 0.3) * 255).astype(np.uint8), "L")
    dm = dilate(dm, args.dark_grow)
    din = np.asarray(running).astype(np.float32)
    if args.dark_noise > 0:
        din += np.random.default_rng(seed).normal(0, args.dark_noise, din.shape) \
            * (np.asarray(dm, np.float32)[..., None] / 255.0)
    din_img = Image.fromarray(np.clip(din, 0, 255).astype(np.uint8), "RGB")
    _, trig = STITCH["satin"]
    pipe.set_adapters(["satin"], [strength])
    out = fill_stage(pipe, args.model, din_img, dm, args.prompt_dark.format(trig=trig),
                     size, steps, guidance, seed, denoise=denoise)
    return Image.composite(out, running, feather(dm, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="the image (required unless --job)")
    ap.add_argument("--masks", help="dir with <stem>__<stitch>.png from paint_stitch_masks.py "
                                    "(required unless --job)")
    ap.add_argument("--job", default=None,
                    help="a job bundle (.zip, folder, or job.json; see JOB_FORMAT.md) instead of "
                         "--image/--masks: one stage per stitch, each layer's own procedural init "
                         "(implies --tex_proc), per-layer prompt/denoise; its settings become "
                         "defaults that flags on the command line override")
    ap.add_argument("--model",
                    choices=["sdxl_base", "sdxl_inpaint", "flux_base", "flux_fill", "qwen_edit"],
                    default="sdxl_base",
                    help="sdxl_base = SDXL masked img2img (LoRA-strong, fast); "
                         "sdxl_inpaint = context-strong (copies surroundings, weak LoRA); "
                         "flux_base = FLUX.1-dev img2img (LoRA-strong); "
                         "flux_fill = FLUX.1-Fill (context-strong, weak LoRA); "
                         "qwen_edit = best quality (fp8)")
    ap.add_argument("--order", default="french_knot,silk_purl,satin",
                    help="stage order (each stage sees the previous output as context); silk_purl "
                         "before satin so it isn't fed freshly-generated satin to copy")
    ap.add_argument("--strength", type=float, default=1.0, help="LoRA strength each stage")
    ap.add_argument("--lora_variant", default="v1",
                    help="LoRA family: v1 (paper) or trigonly_v2 (trigger-only captions, flux1)")
    ap.add_argument("--lora_step", type=int, default=0,
                    help="pick a step checkpoint (e.g. 3000) instead of the final save")
    ap.add_argument("--denoise", type=float, default=1.0,
                    help="img2img denoising strength (flux_base/sdxl); <1 keeps some of the init")
    ap.add_argument("--paste_tex", action="store_true",
                    help="before each stage, tile a crop of real stitch texture over the mask as "
                         "the init (use with --denoise < 1); sources from --tex_<stitch> or the "
                         "first png in data_stitches/<stitch>_trigonly")
    ap.add_argument("--tex_french_knot", default=None,
                    help="texture image(s) for --paste_tex; comma list allowed (see --tex_shuffle)")
    ap.add_argument("--tex_satin", default=None)
    ap.add_argument("--tex_silk_purl", default=None)
    ap.add_argument("--tex_shuffle", action="store_true",
                    help="per run, pick the paste texture at random from the comma list and "
                         "randomise the crop position per region (seeded by the run seed)")
    ap.add_argument("--tex_proc", action="store_true",
                    help="synthesise the paste texture procedurally (bump lattice / thread ridges / "
                         "coil rows) instead of cropping a photo tile; implies --paste_tex. "
                         "A --tex_<stitch> file overrides it for that stitch, and the literal "
                         "value 'proc' in a --tex_<stitch> list selects procedural per stitch")
    ap.add_argument("--proc_knot_r", type=int, default=0,
                    help="procedural knot radius in working-image px (0 = random 11-15)")
    ap.add_argument("--proc_satin_thread", type=float, default=0.0,
                    help="procedural satin thread width in working-image px (0 = default 4.5)")
    ap.add_argument("--proc_satin_angle", type=float, default=0.0,
                    help="procedural satin thread angle in degrees (0 = vertical)")
    ap.add_argument("--proc_satin_angle_step", type=float, default=0.0,
                    help="extra angle per connected satin region, so separate regions "
                         "get differently angled threads (e.g. 30)")
    ap.add_argument("--proc_coil", type=float, default=0.0,
                    help="procedural purl coil pitch in working-image px (0 = random 6-8)")
    ap.add_argument("--proc_col_french_knot", default=None,
                    help="procedural base colour override 'R,G,B' (0-255); pair with a matching "
                         "--prompt_<stitch> so the prompt's colour words agree")
    ap.add_argument("--proc_col_satin", default=None)
    ap.add_argument("--proc_col_silk_purl", default=None)
    ap.add_argument("--proc_noise", type=float, default=0.0,
                    help="procedural texture noise sigma override (0 = per-stitch default: "
                         "8 for knot/satin, 10 for purl)")
    ap.add_argument("--brim", type=int, default=0,
                    help="dilate the mask the MODEL is asked to fill by this many px, so stitches "
                         "end naturally in a band around the region instead of being cut at the "
                         "border; the fill is composited back through the same grown band (minus "
                         "the other stitches' regions, hard border). Supersedes --edge_grow")
    ap.add_argument("--edge_grow", type=int, default=0,
                    help="composite the fill back through a mask dilated this many px, keeping "
                         "the model's own blended edge just past the border; the other stitches' "
                         "regions are subtracted from the grown band so it never eats a sibling "
                         "fill (safe version of filling ~5px outside the edge)")
    ap.add_argument("--stamp_dark", type=float, default=0.0,
                    help="after filling, darken the fill through the original's dark-feature "
                         "alpha (same detection as --keep_dark) by this strength (0-1), so "
                         "drawn design lines/windows read as darker stitching; survives any "
                         "denoise, unlike --keep_dark alone")
    ap.add_argument("--dark_stitch", default="0",
                    help="after all fills (and the --stamp_dark darkening), re-render the dark "
                         "drawn lines/windows as STITCHED features: one extra satin-LoRA masked "
                         "img2img pass over just the (dilated) dark-feature mask at this denoise "
                         "(comma list allowed with --dark_from to ablate; 0 = off; try 0.4-0.55)")
    ap.add_argument("--dark_grow", type=int, default=3,
                    help="dilate the dark-feature mask by this many px before the --dark_stitch "
                         "pass, so a drawn line is about one satin-thread wide")
    ap.add_argument("--dark_noise", type=float, default=6.0,
                    help="noise sigma added inside the dark mask before the --dark_stitch pass, "
                         "so the denoise invents thread structure instead of keeping flat dark")
    ap.add_argument("--prompt_dark", default="{trig}, thin dark brown stitched outline, "
                    "narrow single-thread dark {trig} lines and small dark {trig} blocks",
                    help="prompt for the --dark_stitch pass; {trig} = the satin trigger")
    ap.add_argument("--dark_from", default=None,
                    help="skip generation: load this finished composite, apply --stamp_dark then "
                         "the --dark_stitch pass to it, and save (finishing an already-picked "
                         "output without regenerating the fills)")
    ap.add_argument("--keep_dark", type=float, default=0.0,
                    help="blend the original's pixels darker than this luminance (0-255) back "
                         "over the paste-init inside the mask, so drawn design lines and window "
                         "patches survive into the fill (0 = off; try 100-120)")
    ap.add_argument("--sib_feather", type=int, default=0,
                    help="soften only the boundary between two stitches' fills by this many px "
                         "(alpha blend near sibling regions); the outer edge stays hard")
    ap.add_argument("--region_variants", type=int, default=0,
                    help="--per_region: generate this many seed-variants PER REGION, saving each "
                         "as an RGBA patch under <run>/variants/ plus regions.json, for the "
                         "click-to-cycle picker; variant 0 is composited into the final")
    ap.add_argument("--edge_feather", type=int, default=0,
                    help="feather the composite mask by this many px when pasting fills back, "
                         "so boundaries blend instead of showing a hard cut-out edge")
    ap.add_argument("--region_max_up", type=float, default=0.0,
                    help="--per_region: cap the region upscale factor (e.g. 2 = at most 2x); "
                         "0 = always upscale to the full working size (can shrink stitch scale "
                         "in small regions)")
    ap.add_argument("--denoise_french_knot", type=float, default=None, help="per-stitch denoise override")
    ap.add_argument("--denoise_satin", type=float, default=None)
    ap.add_argument("--denoise_silk_purl", type=float, default=None)
    ap.add_argument("--per_region", action="store_true",
                    help="infill each connected mask region independently: crop a padded square "
                         "around it, upscale to the working size, fill, downscale and paste back. "
                         "Stops structures 'continuing' between disconnected regions and raises "
                         "the effective resolution of small regions")
    ap.add_argument("--per_layer", action="store_true",
                    help="with --per_region: split each stage by layer before finding connected "
                         "regions, so touching layers of one stitch are filled separately")
    ap.add_argument("--proc_base", action="store_true",
                    help="with --paste_tex: paste every stage's init before the first pass, so "
                         "no stage sees the original photo inside any mask (only unmasked pixels "
                         "keep it)")
    ap.add_argument("--paste_only", action="store_true",
                    help="write the pure pasted-init composites (implies --paste_tex) and exit "
                         "without loading any model — for approving the paste before denoising")
    ap.add_argument("--mask_grow", type=int, default=0,
                    help="dilate each mask by this many px before filling; pushes the latent-space "
                         "blend band outside the region so the texture reaches the drawn edge")
    ap.add_argument("--strength_french_knot", type=float, default=None, help="per-stitch LoRA-strength override")
    ap.add_argument("--strength_satin", type=float, default=None)
    ap.add_argument("--strength_silk_purl", type=float, default=None)
    ap.add_argument("--prompt_french_knot", default=None, help="override template; use {trig} for the token")
    ap.add_argument("--prompt_satin", default=None)
    ap.add_argument("--prompt_silk_purl", default=None)
    ap.add_argument("--blend_rim", type=int, default=0,
                    help="sdxl only: after the base img2img fill, re-run SDXL-inpaint (same LoRA) on "
                         "a boundary band this many px wide to blend the seam (0 = off; try 24-48)")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seeds", default=None,
                    help="comma list of one seed PER STAGE in --order (e.g. 1234,1240,1240) to "
                         "hand-mix the best seed for each stitch; overrides --tries/--seed")
    ap.add_argument("--size", type=int, default=0, help="0 = model default")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--transformer_file", default=SF_QEDIT, help="qwen_edit single-file transformer")
    ap.add_argument("--no_fp8", action="store_true", help="qwen_edit: skip fp8 (needs ~40GB VRAM)")
    ap.add_argument("--quant_cache", default=None,
                    help="qwen_edit: torch.save/load path for the assembled fp8+LoRA transformer "
                         "(skips the ~10-min quantise on later runs; delete if LoRAs/order change)")
    ap.add_argument("--out", default="staged_out")
    # --job: two-stage parse so the job's settings become defaults and the CLI still wins
    job, jp = None, ap.parse_known_args()[0].job
    if jp:
        job = load_job(jp)
        for k in job.get("stitches") or {}:             # per-stitch flags for its own stitches
            for f, t in (("denoise", float), ("strength", float), ("prompt", str)):
                if f"--{f}_{k}" not in ap._option_string_actions:
                    ap.add_argument(f"--{f}_{k}", type=t, default=None)
        ap.set_defaults(**{"tries": 1, **job_settings(ap, job)})   # one run per job unless asked
    args = ap.parse_args()
    if job is None and not (args.image and args.masks):
        ap.error("--image and --masks are required (or give --job)")

    pcol = {}
    for st_ in STITCH:
        v = getattr(args, f"proc_col_{st_}")
        if v:
            pcol[st_] = tuple(int(c) for c in v.split(","))

    ds_list = [float(v) for v in str(args.dark_stitch).split(",") if v]
    ds_on = any(v > 0 for v in ds_list)
    if job is None and ds_on and "satin" not in args.order:
        raise SystemExit("--dark_stitch needs the satin LoRA: include satin in --order")

    if job is None:
        img_path = Path(args.image)
        stem = img_path.stem
        order = [s for s in args.order.split(",") if s]
        masks_dir = Path(args.masks)
    else:
        if args.dark_from:
            raise SystemExit("--dark_from is not supported with --job (run it on the legacy CLI)")
        img_path, stem = Path(job["image"]), job["name"]
        for k, s in (job.get("stitches") or {}).items():
            STITCH[k] = (k, s["trigger"])
            LORA_FILE[k] = s["lora"]
        seen = list(dict.fromkeys(L["stitch"] for L in job["layers"]))
        want = [s for s in args.order.split(",") if s]  # job order (or --order); rest after
        order = [s for s in want if s in seen] + [s for s in seen if s not in want]
        if ds_on and "satin" not in order:
            raise SystemExit("--dark_stitch needs the satin LoRA: the job has no satin layer")
        args.tex_proc = True

    if args.paste_only or args.tex_proc:
        args.paste_tex = True

    # layers: legacy = one per stitch (its mask, the global --proc_* look); --job = the
    # job's layers, each with its own colour/shape and optional prompt/denoise
    if job is None:
        layers = [dict(id=st, stitch=st, init=st, mask=masks_dir / f"{stem}__{st}.png",
                       col=pcol.get(st)) for st in order]
    else:
        sdef = job.get("stitches") or {}
        layers = [dict(id=L["id"], stitch=L["stitch"], mask=L["mask"],
                       init=sdef[L["stitch"]]["init"] if L["stitch"] in sdef else L["stitch"],
                       col=hex_rgb(L.get("colour")) or pcol.get(L["stitch"]),
                       angle=L.get("angle"), thread=L.get("thread"), knot_r=L.get("knot_r"),
                       coil=L.get("coil"), prompt=L.get("prompt"), denoise=L.get("denoise"))
                  for L in job["layers"]]
    stage_layers = {st: [i for i, L in enumerate(layers) if L["stitch"] == st] for st in order}

    # resolve masks + LoRAs up front so we fail before loading a 20B model
    masks, loras = {}, {}
    for st in order:
        mp = layers[stage_layers[st][0]]["mask"]
        if not Path(mp).exists():
            raise SystemExit(f"missing mask: {mp}")
        masks[st] = mp
        if not args.paste_only:
            lp = lora_path(st, args.model, args.lora_variant, args.lora_step)
            if not Path(lp).exists():
                raise SystemExit(f"missing LoRA: {lp}")
            loras[st] = lp

    # resolve paste textures up front too
    tex = {}
    if args.paste_tex:
        for st in order:
            tp = getattr(args, f"tex_{st}", None)
            if tp is None:
                if args.tex_proc:
                    tex[st] = ["proc"]
                    print(f"paste texture(s) for {st}: procedural")
                    continue
                cands = sorted(Path(f"data_stitches/{st}_trigonly").glob("*.png")) + \
                        sorted(Path(f"data_stitches/{st}_trigonly").glob("*.jpg"))
                if not cands:
                    raise SystemExit(f"--paste_tex: no texture found for {st}; pass --tex_{st}")
                tp = str(cands[0])
            paths = [p for p in tp.split(",") if p]
            for p in paths:
                if p != "proc" and not Path(p).exists():
                    raise SystemExit(f"missing texture: {p}")
            tex[st] = paths
            print(f"paste texture(s) for {st}: {paths}")

    W0, H0 = Image.open(img_path).size

    def crop_box(size):
        """--job: where the image sits in the padded size x size square (None = legacy,
        the image is squashed to the square as before)"""
        if job is None:
            return None
        s = size / max(W0, H0)
        w, h = max(1, round(W0 * s)), max(1, round(H0 * s))
        x0, y0 = (size - w) // 2, (size - h) // 2
        return (x0, y0, x0 + w, y0 + h)

    def to_square(im, size, resample, pad):
        """legacy: resize to size x size. --job: long side -> size, then pad to the
        square (pad='reflect' for the image, 'constant' = zeros for masks)"""
        cb = crop_box(size)
        if cb is None:
            return im.resize((size, size), resample)
        a = np.asarray(im.resize((cb[2] - cb[0], cb[3] - cb[1]), resample))
        pw = ((cb[1], size - cb[3]), (cb[0], size - cb[2])) + ((0, 0),) * (a.ndim - 2)
        return Image.fromarray(np.pad(a, pw, mode=pad), im.mode)

    def load_base(size):
        return to_square(Image.open(img_path).convert("RGB"), size, Image.LANCZOS, "reflect")

    def load_mask(path, size):
        m = to_square(Image.open(path).convert("L"), size, Image.NEAREST, "constant")
        if args.mask_grow > 0:
            from PIL import ImageFilter
            m = m.filter(ImageFilter.MaxFilter(2 * args.mask_grow + 1))
            m = m.point(lambda v: 255 if v >= 128 else 0)
        return m

    def load_layers(size):
        """per-layer masks + per-stage unions. --job: later layers win on overlap
        (whatever the stitch), so every pixel belongs to at most one layer"""
        lm = [load_mask(L["mask"], size) for L in layers]
        if job is None:
            return lm, {st: lm[stage_layers[st][0]] for st in order}
        taken = np.zeros((size, size), bool)
        for i in reversed(range(len(lm))):
            a = np.asarray(lm[i]) >= 128
            lm[i] = Image.fromarray(((a & ~taken) * 255).astype(np.uint8), "L")
            taken |= a
        return lm, {st: Image.fromarray((np.any([np.asarray(lm[i]) >= 128 for i in stage_layers[st]],
                                                axis=0) * 255).astype(np.uint8), "L")
                    for st in order}

    def layer_src(L, prng):
        """procedural paste source for one layer: its own colour/shape, --proc_* as fallback"""
        def mk(k):
            a0 = args.proc_satin_angle if L.get("angle") is None else L["angle"]
            ang = (a0 + k * args.proc_satin_angle_step) \
                if L["init"] == "satin" else 0.0
            return proc_tex(L["init"], prng, knot_r=L.get("knot_r") or args.proc_knot_r,
                            satin_thread=L.get("thread") or args.proc_satin_thread,
                            coil=L.get("coil") or args.proc_coil, noise=args.proc_noise,
                            col=L["col"], angle=ang)
        return mk if (L["init"] == "satin" and args.proc_satin_angle_step) else mk(0)

    def paste_stage(running, st, lm, prng, rng):
        """paste-init each layer of stitch `st` into its own mask -> (image, texture used)"""
        tp = rng.choice(tex[st]) if rng else tex[st][0]
        for i in stage_layers[st]:
            running = paste_init(running, lm[i], layer_src(layers[i], prng) if tp == "proc" else tp,
                                 rng=rng)
        return running, tp

    def cut(im):
        """--job: crop a working-square image back to the image's own frame"""
        return im if cb is None else im.crop(cb)

    def save_job(out_root):
        if job is not None:
            import shutil
            out_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(job["_file"], out_root / "job.json")

    if args.paste_only:
        # no model: just apply the pastes stage by stage and write the previews
        size = round16(args.size or 1024)
        cb = crop_box(size)
        base = load_base(size)
        lm, mL = load_layers(size)
        tdir = Path(args.out) / stem / "paste_preview"
        if args.mask_grow > 0:
            tdir = tdir.with_name(f"paste_preview_grow{args.mask_grow}")
        tdir.mkdir(parents=True, exist_ok=True)
        save_job(tdir.parent)
        import random
        rng = random.Random(args.seed) if args.tex_shuffle else None
        running = base.copy()
        prng = rng or random.Random(args.seed)
        for i, st in enumerate(order, 1):
            running, _ = paste_stage(running, st, lm, prng, rng)
            cut(running).save(tdir / f"stage{i}_{st}_pasted.png")
        b_, r_ = np.asarray(cut(base)), np.asarray(cut(running))
        strip = np.concatenate([b_, np.full((b_.shape[0], 6, 3), 200, np.uint8), r_], axis=1)
        Image.fromarray(strip).save(tdir / "paste_strip.png")
        print(f"paste-only previews -> {tdir}")
        return

    if args.model.startswith("sdxl"):
        base_id = SDXL_BASE_ID if args.model == "sdxl_base" else SDXL_INPAINT_ID
        pipe, dflt = build_sdxl(loras, base_id)
    elif args.model == "flux_fill":
        pipe, dflt = build_flux_fill(loras)
    elif args.model.startswith("flux"):
        pipe, dflt = build_flux(loras)
    else:
        pipe, dflt = build_qwen_edit(loras, args.transformer_file, fp8=not args.no_fp8,
                                     quant_cache=args.quant_cache)

    pipe_rim = None
    if args.blend_rim > 0:
        if args.model.startswith("sdxl"):
            print("Loading SDXL-inpaint for rim seam-blending …")
            pipe_rim, _ = build_sdxl(loras, SDXL_INPAINT_ID)
        else:
            print("WARN: --blend_rim is only wired for sdxl models; ignoring.")
    size = round16(args.size or dflt["size"])
    steps = args.steps or dflt["steps"]
    guidance = args.guidance or dflt["guidance"]

    cb = crop_box(size)
    base = load_base(size)
    lm, mL = load_layers(size)
    lma = [np.asarray(m) >= 128 for m in lm]
    qwen = args.model.startswith("qwen")

    def layer_pd(L):
        """(prompt, denoise) for one layer. --job: layer > --prompt_/--denoise_<stitch>
        (flag or job setting) > the job's stitch prompt > JOB_PROMPT, {colour} filled in"""
        st = L["stitch"]
        _, trig = STITCH[st]
        if job is None:
            tmpl = getattr(args, f"prompt_{st}") or (QWEN_PROMPT[st] if qwen else PROMPT[st])
            prompt = tmpl.format(trig=trig)
        else:
            jp_ = JOB_QWEN_PROMPT if qwen else JOB_PROMPT
            tmpl = L.get("prompt") or getattr(args, f"prompt_{st}", None) or \
                (job.get("stitches") or {}).get(st, {}).get("prompt") or jp_.get(st, jp_["custom"])
            prompt = tmpl.format(trig=trig, colour=colour_name(L["col"] or DEF_COL[L["init"]]))
        dn = L.get("denoise")
        if dn is None:
            dn = getattr(args, f"denoise_{st}", None)
        return prompt, (float(dn) if dn is not None else args.denoise)

    def majority(st, region):
        """the stitch's layer covering most of `region` (bool array)"""
        ids = stage_layers[st]
        return ids[int(np.argmax([(lma[i] & region).sum() for i in ids]))]
    # --stamp_dark: fixed darkening alpha from the original's dark features in any mask
    stamp_alpha = None
    if args.stamp_dark > 0 or ds_on:
        from PIL import ImageFilter
        un = np.any([np.asarray(mL[s_]) >= 128 for s_ in order], axis=0)
        lum = np.asarray(base).astype(np.float32).mean(2)
        sa = np.clip(((args.keep_dark or 110) - lum) / 40.0, 0, 1)
        sa[~un] = 0
        stamp_alpha = np.asarray(Image.fromarray((sa * 255).astype(np.uint8), "L")
                                 .filter(ImageFilter.GaussianBlur(1))).astype(np.float32) / 255.0

    # per stitch: union of the OTHER stitches' masks (protected zone for --edge_grow)
    others = {}
    for st in order:
        rest = [np.asarray(mL[o]) >= 128 for o in order if o != st]
        others[st] = Image.fromarray((np.any(rest, axis=0) * 255).astype(np.uint8), "L") \
            if rest else None

    if args.dark_from:
        # finishing mode: stamp + dark-stitch pass on an existing composite, no fills
        if not ds_on:
            raise SystemExit("--dark_from needs --dark_stitch > 0")
        s_sat = getattr(args, "strength_satin")
        s_sat = float(s_sat) if s_sat is not None else float(args.strength)
        running0 = Image.open(args.dark_from).convert("RGB").resize((size, size), Image.LANCZOS)
        if args.stamp_dark > 0:
            running0 = Image.fromarray(np.clip(
                np.asarray(running0).astype(np.float32)
                * (1 - args.stamp_dark * stamp_alpha[..., None]), 0, 255).astype(np.uint8), "RGB")
        tdir = Path(args.out) / stem / \
            f"darkfrom_{Path(args.dark_from).stem}_dg{args.dark_grow}_seed{args.seed}"
        tdir.mkdir(parents=True, exist_ok=True)
        running0.save(tdir / "stamped_input.png")
        for v in ds_list:
            out = dark_stitch_pass(pipe, args, running0, stamp_alpha, size, steps,
                                   guidance, args.seed + 77, v, s_sat)
            out.save(tdir / f"final_ds{v:g}.png")
            print(f"  dark-stitch ds={v:g} -> {tdir / f'final_ds{v:g}.png'}")
        print(f"\nDone -> {tdir}")
        return

    # build the runs: each run is (per-stage seed list, tag). --seeds = one hand-mixed
    # run with a seed per stage; otherwise --tries runs, each one seed for all stages.
    if args.seeds:
        slist = [int(s) for s in args.seeds.split(",")]
        if len(slist) != len(order):
            raise SystemExit(f"--seeds has {len(slist)} values but --order has {len(order)} stages")
        runs = [(slist, "seeds" + "-".join(map(str, slist)))]
    else:
        runs = [([args.seed + t] * len(order), f"seed{args.seed + t}") for t in range(args.tries)]

    variant_tag = "" if args.lora_variant == "v1" else f"_{args.lora_variant}"
    if args.lora_step > 0:
        variant_tag += f"@{args.lora_step}"
    if args.denoise != 1.0:
        variant_tag += f"_dn{args.denoise:g}"
    if args.paste_tex:
        variant_tag += "_procpaste" if args.tex_proc else "_paste"
    if args.per_region:
        variant_tag += "_perlay" if args.per_layer else "_perreg"
    if args.brim > 0:
        variant_tag += f"_brim{args.brim}"
    if args.sib_feather > 0:
        variant_tag += f"_sf{args.sib_feather}"
    if args.keep_dark > 0:
        variant_tag += f"_kd{args.keep_dark:g}"
    if args.stamp_dark > 0:
        variant_tag += f"_sd{args.stamp_dark:g}"
    if ds_on:
        variant_tag += f"_ds{ds_list[0]:g}_dg{args.dark_grow}"
    if args.edge_grow > 0:
        variant_tag += f"_eg{args.edge_grow}"
    if args.edge_feather > 0:
        variant_tag += f"_f{args.edge_feather}"
    if args.region_max_up > 0:
        variant_tag += f"_up{args.region_max_up:g}"
    if args.mask_grow > 0:
        variant_tag += f"_grow{args.mask_grow}"
    if args.proc_satin_angle or args.proc_satin_angle_step:
        variant_tag += f"_sa{args.proc_satin_angle:g}s{args.proc_satin_angle_step:g}"

    if args.proc_base and args.paste_tex:
        variant_tag += "_pbase"
    out_root = Path(args.out) / stem
    save_job(out_root)
    for run_i, (seeds, tag) in enumerate(runs, 1):
        tdir = out_root / f"{args.model}{variant_tag}_{tag}"
        tdir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== run {run_i}/{len(runs)}  seeds={seeds}  ({args.model}) ===")
        import random
        rng = random.Random(seeds[0]) if args.tex_shuffle else None
        tex_used = {}
        variants_meta = []
        running = base.copy()
        if args.proc_base and args.paste_tex:
            # --proc_base: every mask gets its init up front; each stage re-pastes its own
            _brng = rng or random.Random(seeds[0])
            for st in order:
                running, _ = paste_stage(running, st, lm, _brng, rng)
            cut(running).save(tdir / "proc_base.png")
        for i, st in enumerate(order, 1):
            seed = seeds[i - 1]                                  # per-stage seed
            _, trig = STITCH[st]
            # prompt/denoise per layer; a stage-wide pass uses the layer covering most of it
            pd = {li: layer_pd(layers[li]) for li in stage_layers[st]}
            prompt, dn = pd[majority(st, np.asarray(mL[st]) >= 128)]
            s_st = getattr(args, f"strength_{st}", None)
            s_st = float(s_st) if s_st is not None else float(args.strength)
            pipe.set_adapters([st], [s_st])
            stage_in = running
            # --brim: the model fills a band past the border too, and the same grown
            # band (minus siblings) is what gets pasted back
            bgrow = args.brim if args.brim > 0 else args.edge_grow
            if args.paste_tex:
                stage_in, tp = paste_stage(running, st, lm, rng or random.Random(seed), rng)
                tex_used[st] = "proc" if tp == "proc" else os.path.basename(tp)
                if args.keep_dark > 0:
                    stage_in = keep_dark(stage_in, base, mL[st], args.keep_dark)
                cut(stage_in).save(tdir / f"stage{i}_{st}_pasted_init.png")
            if args.per_region:
                # each connected region independently: crop a padded square, upscale to
                # the working size, fill, downscale, paste back through the region mask
                import cv2
                # --per_layer: split by layer first, so adjacent layers of one stitch
                # (e.g. the tones of a face) are filled separately, each in its own crop
                srcs = [lma[li] for li in stage_layers[st]] if args.per_layer \
                    else [np.asarray(mL[st]) >= 128]
                comps = []
                for sm in srcs:
                    n_comp, labels = cv2.connectedComponents(sm.astype(np.uint8))
                    comps += [labels == c for c in range(1, n_comp)]
                done = 0
                for lab, comp in enumerate(comps, 1):
                    if comp.sum() < 25:                  # specks: the paste-init already fills them
                        continue
                    ys, xs = np.nonzero(comp)
                    l, u, r_, lo = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                    side = max(r_ - l, lo - u)
                    side = min(size, side + 2 * max(32, side // 4))
                    x0 = int(np.clip((l + r_) // 2 - side // 2, 0, size - side))
                    y0 = int(np.clip((u + lo) // 2 - side // 2, 0, size - side))
                    box = (x0, y0, x0 + side, y0 + side)
                    rsize = size if args.region_max_up <= 0 else \
                        round16(min(size, max(256, int(side * args.region_max_up))))
                    comp_L = Image.fromarray((comp * 255).astype(np.uint8), "L")
                    crop_in = stage_in.crop(box).resize((rsize, rsize), Image.LANCZOS)
                    crop_m = dilate(comp_L, args.brim).crop(box).resize((rsize, rsize), Image.NEAREST)
                    cm = grow_composite(comp_L, others[st], bgrow, args.edge_feather,
                                        sib=args.sib_feather)
                    cma = np.asarray(cm)
                    pys, pxs = np.nonzero(cma > 0)
                    pbox = (int(pxs.min()), int(pys.min()), int(pxs.max()) + 1, int(pys.max()) + 1)
                    if cb is not None:                   # --job: clip the patch to the image
                        pbox = (max(pbox[0], cb[0]), max(pbox[1], cb[1]),
                                min(pbox[2], cb[2]), min(pbox[3], cb[3]))
                    li = majority(st, comp)              # prompt/denoise from the majority layer
                    prompt_c, dn_c = pd[li]
                    if job is not None:
                        print(f"    region {lab}: layer {layers[li]['id']} dn={dn_c:g} :: {prompt_c}")
                    first = None
                    vfiles = []
                    for k in range(max(1, args.region_variants)):
                        out_c = fill_stage(pipe, args.model, crop_in, crop_m, prompt_c, rsize, steps,
                                           guidance, seed + 1000 * lab + k, denoise=dn_c)
                        patched = running.copy()
                        patched.paste(out_c.resize((side, side), Image.LANCZOS), (x0, y0))
                        if first is None:
                            first = patched
                        if args.region_variants > 0:
                            vdir = tdir / "variants"
                            vdir.mkdir(exist_ok=True)
                            pa = np.asarray(patched)[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                            if stamp_alpha is not None:
                                sac = stamp_alpha[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                                pa = np.clip(pa * (1 - args.stamp_dark * sac[..., None]),
                                             0, 255).astype(np.uint8)
                            al = cma[pbox[1]:pbox[3], pbox[0]:pbox[2]]
                            fn = f"{st}_r{lab}_v{k}.png"
                            Image.fromarray(np.dstack([pa, al]), "RGBA").save(vdir / fn)
                            vfiles.append(fn)
                    if args.region_variants > 0:
                        variants_meta.append({"stitch": st, "region": int(lab),
                                              "bbox": list(pbox), "files": vfiles})
                        if cb is not None:               # bbox in the cropped image's frame
                            variants_meta[-1]["bbox"] = [pbox[0] - cb[0], pbox[1] - cb[1],
                                                         pbox[2] - cb[0], pbox[3] - cb[1]]
                            variants_meta[-1]["layer"] = layers[li]["id"]
                    running = Image.composite(first, running, cm)
                    done += 1
                print(f"    per-region: {done} regions filled")
            else:
                out = fill_stage(pipe, args.model, stage_in, dilate(mL[st], args.brim),
                                 prompt, size, steps, guidance, seed, denoise=dn)
                cm = grow_composite(mL[st], others[st], bgrow, args.edge_feather,
                                    sib=args.sib_feather)
                running = Image.composite(out, running, cm)      # body: only this region changes
            if pipe_rim is not None:                             # seam pass: blend the boundary band
                rim = rim_mask(mL[st], args.blend_rim)
                pipe_rim.set_adapters([st], [s_st])
                out2 = fill_stage(pipe_rim, "sdxl", running, rim, prompt, size, steps, guidance, seed)
                running = Image.composite(out2, running, rim)
            cut(running).save(tdir / f"stage{i}_{st}.png")
            print(f"  stage {i}: {st} ({trig}) seed={seed}" + ("  +rim" if pipe_rim is not None else ""))
        if stamp_alpha is not None:
            running = Image.fromarray(np.clip(
                np.asarray(running).astype(np.float32)
                * (1 - args.stamp_dark * stamp_alpha[..., None]), 0, 255).astype(np.uint8), "RGB")
        if ds_on:
            s_sat = getattr(args, "strength_satin")
            s_sat = float(s_sat) if s_sat is not None else float(args.strength)
            running = dark_stitch_pass(pipe, args, running, stamp_alpha, size, steps,
                                       guidance, seeds[0] + 77, ds_list[0], s_sat)
            cut(running).save(tdir / "dark_stitch.png")
        if variants_meta:
            import json
            cut(base).save(tdir / "variants" / "base.png")
            rj = {"size": size, "order": order, "regions": variants_meta}
            if cb is not None:                                   # --job: non-square frame
                rj = {"size": size, "width": cb[2] - cb[0], "height": cb[3] - cb[1],
                      "order": order, "regions": variants_meta}
            vj = json.dumps(rj, indent=1)
            (tdir / "variants" / "regions.json").write_text(vj)
            # regions.js lets the picker page work from file:// (fetch is blocked there)
            (tdir / "variants" / "regions.js").write_text("window.REGIONS = " + vj + ";")
            print(f"  variants: {sum(len(v['files']) for v in variants_meta)} patches, "
                  f"{len(variants_meta)} regions -> {tdir / 'variants'}")
        md = ({"model": args.model, "order": ",".join(order), "strength": args.strength,
                         "steps": steps, "guidance": guidance, "size": size,
                         "lora_variant": args.lora_variant, "lora_step": args.lora_step,
                         "denoise": args.denoise, "paste_tex": int(args.paste_tex),
                         "tex_proc": int(args.tex_proc), "per_region": int(args.per_region),
                         "brim": args.brim, "proc_noise": args.proc_noise,
                         "sib_feather": args.sib_feather, "region_variants": args.region_variants,
                         "keep_dark": args.keep_dark, "stamp_dark": args.stamp_dark,
                         "edge_grow": args.edge_grow,
                         "edge_feather": args.edge_feather, "region_max_up": args.region_max_up,
                         "denoise_per": ";".join(
                             f"{s}={getattr(args, f'denoise_{s}', None) if getattr(args, f'denoise_{s}', None) is not None else args.denoise:g}"
                             for s in order),
                         "mask_grow": args.mask_grow,
                         "tex_used": ";".join(f"{k}={v}" for k, v in tex_used.items()),
                         "seeds": ",".join(map(str, seeds))})
        if job is not None:
            md["job"] = stem
            md["layers"] = ";".join(f"{L['id']}={L['stitch']}" for L in layers)
        meta = png_meta(md)
        cut(running).save(tdir / "final.png", pnginfo=meta)
        # strip: original | all-masks overlay | final
        ov = np.asarray(base).copy()
        for st, col in zip(order, [(31, 119, 180), (44, 160, 44), (214, 39, 40),
                                   (148, 103, 189), (255, 127, 14), (23, 190, 207)]):
            m = np.asarray(mL[st]) >= 128
            ov[m] = (0.45 * ov[m] + 0.55 * np.array(col)).astype(np.uint8)
        b_, o_, r_ = (np.asarray(cut(x)) for x in (base, Image.fromarray(ov), running))
        sep = np.full((b_.shape[0], 6, 3), 200, np.uint8)
        strip = np.concatenate([b_, sep, o_, sep, r_], axis=1)
        Image.fromarray(strip).save(tdir / "strip.png")
        print(f"  -> {tdir}\\final.png")

    print(f"\nDone -> {out_root}")


if __name__ == "__main__":
    main()
