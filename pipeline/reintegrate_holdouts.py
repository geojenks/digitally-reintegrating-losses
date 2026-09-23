"""
Reintegrate losses in holdout embroidery by LoRA-conditioned inpainting, sweeping
LoRA strength as a built-in baseline.

The LoRA-strength sweep (0, 0.25, 0.5, 0.75, 1.0) holds the base model, mask,
prompt and seed fixed and varies only the adapter, so any change in the fill is
attributable solely to the learned stitch prior. Strength 0 == the base model
with no craft prior (the baseline); the hypothesis is that fidelity to the
target stitch improves monotonically with strength — most visibly in the
heterogeneous-context case, where the surround is a *different* stitch and the
base model has no contextual cue to copy.

Currently implements FLUX.1-Fill-dev (the primary quantitative model). SDXL-
inpainting and Qwen-Image-Edit slot into the MODELS registry the same way (see
TRAINING_TO_INFERENCE.txt for the train->inference mapping).

Pipeline position: this produces RGB fills. To score them, run the fine-tuned
Marigold over the fills (generate_type_normals.py pattern) then
texture_fidelity.py sweep / compare, restricting to the fill region via the mask.

Mask convention (matches generate_synthetic_masks.py and FLUX.1-Fill): WHITE =
region to regenerate, BLACK = keep.

Run with the Marigold venv (diffusers 0.38, FluxFillPipeline). FLUX.1-Fill-dev is
gated + ~24 GB; first run downloads it (same HF login as FLUX.1-dev).

Usage
-----
"Marigold training\\venv\\Scripts\\python" reintegrate_holdouts.py ^
    --model     flux1_fill ^
    --images    data_stitches\\holdouts-synthetic_damage\\french_knot\\images ^
    --mask_dir  data_stitches\\holdouts-synthetic_damage\\french_knot\\synthetic_masks\\512x512 ^
    --lora      output\\french_knot_stitch_flux1_lora_v1\\french_knot_stitch_flux1_lora_v1_000003000.safetensors ^
    --trigger   embfnchknt ^
    --strengths 0,0.25,0.5,0.75,1.0 ^
    --size      1024 --steps 50 --guidance 30 ^
    --out       damage_restoration\\flux1\\french_knot\\512x512
"""

import argparse
import os
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def quantize_transformer_fp8(transformer):
    """Quantize a transformer's Linear weights to fp8 (torchao, weight-only).

    A 20B Qwen in bf16 is ~40GB and overflows a 32GB card into shared system
    memory, where every step stalls on PCIe (~40 min/image). fp8 weights halve
    that to ~20GB so it fits in VRAM under model-cpu-offload — removing the spill,
    which is the actual bottleneck. Matmuls upcast to bf16, so quality tracks
    bf16 and the bf16 LoRA + offload hooks keep working.

    Call AFTER load_lora_weights. The LoRA must be injected into ordinary bf16
    Linears first: if the base is already torchao-quantized, peft routes the
    injection through TorchaoLoraLinear/dispatch_torchao, which this peft+torchao
    pairing mis-calls (missing get_apply_tensor_subclass) and crashes. Quantizing
    afterwards, with a filter that skips the adapter's own lora_A/lora_B Linears,
    keeps the adapter bf16 (so per-fill strength scaling still works) while the
    big base weights become fp8.
    """
    import torch.nn as nn
    from torchao.quantization import quantize_, Float8WeightOnlyConfig

    def _base_only(module, fqn: str) -> bool:
        # Quantize base Linears (incl. peft's `base_layer`) but never the small
        # adapter matrices (`lora_A` / `lora_B`), which must stay bf16.
        return isinstance(module, nn.Linear) and "lora_" not in fqn

    print("Quantizing transformer -> fp8 (torchao Float8WeightOnlyConfig) …")
    quantize_(transformer, Float8WeightOnlyConfig(), filter_fn=_base_only)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class Flux1Fill:
    """FLUX.1-Fill-dev mask-based inpainting with a transferred FLUX.1-dev LoRA."""
    base_id = "black-forest-labs/FLUX.1-Fill-dev"
    default_trigger = "embfnchknt"
    default_size, default_steps, default_guidance = 1024, 50, 30.0

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.bfloat16,
                 base_id=None, transformer_file=None, components_id=None):
        from diffusers import FluxFillPipeline
        if transformer_file:
            # Assemble Fill from a local single-file transformer + the cached
            # FLUX.1-dev components (VAE/T5/CLIP/tokenizers/scheduler are identical
            # between dev and Fill — only the transformer differs). No download.
            from diffusers import FluxTransformer2DModel
            cid = components_id or "black-forest-labs/FLUX.1-dev"
            print(f"Loading Fill transformer from single file: {transformer_file}")
            transformer = FluxTransformer2DModel.from_single_file(transformer_file, torch_dtype=dtype)
            print(f"Assembling FluxFillPipeline from cached components: {cid}")
            self.pipe = FluxFillPipeline.from_pretrained(cid, transformer=transformer, torch_dtype=dtype)
        else:
            bid = base_id or self.base_id
            print(f"Loading {bid} …")
            self.pipe = FluxFillPipeline.from_pretrained(bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            # Split into dir + weight_name so it resolves under HF_HUB_OFFLINE=1
            # (offline can't auto-guess the weight filename from a bare path).
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()      # 12B model — offload for VRAM
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            image=image, mask_image=mask,
            height=size, width=size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            max_sequence_length=512,
            generator=gen,
        ).images[0]


class Flux1Inpaint:
    """FLUX.1-dev img2img inpainting (FluxInpaintPipeline) with a native FLUX.1-dev LoRA.

    The 'base' FLUX inpaint path, contrasted with Flux1Fill. Fill blanks the hole
    and regenerates from a 384-ch masked-image conditioning and has NO strength
    knob, so on a large homogeneous hole it fills the void with a centred orb. This
    is plain img2img+mask on the *dev* transformer the LoRA was actually trained on,
    and it EXPOSES `strength` (set via --denoise): at 1.0 it fully regenerates the
    masked region (no GT leak); at <1.0 it keeps a faint seed of the input there, so
    there's no void to orb into. The dev transformer is cached -> no download.
    """
    base_id = "black-forest-labs/FLUX.1-dev"
    default_trigger = "embfnchknt"
    default_size, default_steps, default_guidance = 1024, 28, 7.0
    default_denoise = 1.0

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.bfloat16,
                 base_id=None, transformer_file=None, components_id=None):
        from diffusers import FluxInpaintPipeline
        if transformer_file:
            # Assemble from a local single-file dev transformer + cached components
            # (same FluxTransformer2DModel arch as Fill; only weights differ). No fetch.
            from diffusers import FluxTransformer2DModel
            cid = components_id or self.base_id
            print(f"Loading dev transformer from single file: {transformer_file}")
            transformer = FluxTransformer2DModel.from_single_file(transformer_file, torch_dtype=dtype)
            print(f"Assembling FluxInpaintPipeline from cached components: {cid}")
            self.pipe = FluxInpaintPipeline.from_pretrained(cid, transformer=transformer, torch_dtype=dtype)
        else:
            bid = base_id or self.base_id
            print(f"Loading {bid} …")
            self.pipe = FluxInpaintPipeline.from_pretrained(bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()      # 12B model — offload for VRAM
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.denoise = self.default_denoise           # overridden from --denoise in main
        self.seed_surround = False                    # set from --seed_surround in main

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        img_in = image
        if self.denoise < 1.0 and self.seed_surround and mask is not None:
            img_in = seed_from_surround(image, mask)  # GT-free seed (else <1.0 leaks the answer)
        return self.pipe(
            prompt=prompt,
            image=img_in, mask_image=mask,
            height=size, width=size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=self.denoise,                    # <1 keeps a seed; 1.0 = full regen
            max_sequence_length=512,
            generator=gen,
        ).images[0]


class SdxlInpaint:
    """
    SDXL inpainting with a transferred SDXL-base LoRA.

    Loads the *base* SDXL (already cached) into StableDiffusionXLInpaintPipeline,
    which performs masked img2img when the UNet has 4 input channels — the
    "Set Latent Noise Mask" path in TRAINING_TO_INFERENCE.txt. No dedicated
    inpaint checkpoint download needed; quality is below the inpaint variant but
    fine for validating the workflow. Swap base_id to
    'diffusers/stable-diffusion-xl-1.0-inpainting-0.1' once that is downloaded.
    """
    base_id = "stabilityai/stable-diffusion-xl-base-1.0"
    default_trigger = "embfrench_knot"
    default_size, default_steps, default_guidance = 1024, 40, 7.5

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.float16,
                 base_id=None, transformer_file=None, components_id=None):
        from diffusers import StableDiffusionXLInpaintPipeline
        bid = base_id or self.base_id
        print(f"Loading {bid} …")
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            # Split into dir + weight_name so it resolves under HF_HUB_OFFLINE=1
            # (offline can't auto-guess the weight filename from a bare path).
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            image=image, mask_image=mask,
            height=size, width=size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=1.0,                         # fully regenerate the masked region
            generator=gen,
        ).images[0]


class ZImageInpaint:
    """
    Z-Image mask inpainting with a base-trained Z-Image LoRA, run on Z-Image-Turbo.

    The LoRA trains on the non-turbo base (plain flow-matching) and transfers onto
    Turbo — same ZImageTransformer2DModel — for fast few-step inpainting. Turbo is
    step-distilled, so defaults are few steps + guidance ~1 (no CFG). For the
    slower/heavier base instead, pass --base_id <base> --steps 30 --guidance 4.

    UNTESTED until a trained Z-Image LoRA + the Turbo weights are in place; verify
    the first run (LoRA key format, turbo step/guidance sweet spot).
    """
    base_id = "Tongyi-MAI/Z-Image-Turbo"
    default_trigger = "embfnchknt"
    default_size, default_steps, default_guidance = 1024, 8, 1.0   # distilled-turbo settings

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.bfloat16,
                 base_id=None, transformer_file=None, components_id=None):
        from diffusers import ZImageInpaintPipeline
        if transformer_file:
            # Assemble from a local single-file transformer (e.g. the ComfyUI
            # z_image_turbo_bf16.safetensors) + the cached Z-Image *base* repo's
            # components (Qwen3 text-encoder/VAE/scheduler/tokenizer are shared
            # between base and Turbo — only the transformer differs). Avoids the
            # slow Turbo repo download. Base is complete in cache.
            from diffusers import ZImageTransformer2DModel
            cid = components_id or "Tongyi-MAI/Z-Image"
            print(f"Loading Z-Image transformer from single file: {transformer_file}")
            transformer = ZImageTransformer2DModel.from_single_file(transformer_file, torch_dtype=dtype)
            print(f"Assembling ZImageInpaintPipeline from cached components: {cid}")
            self.pipe = ZImageInpaintPipeline.from_pretrained(cid, transformer=transformer, torch_dtype=dtype)
        else:
            bid = base_id or self.base_id
            print(f"Loading {bid} …")
            self.pipe = ZImageInpaintPipeline.from_pretrained(bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            # Split into dir + weight_name so it resolves under HF_HUB_OFFLINE=1
            # (offline can't auto-guess the weight filename from a bare path).
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            image=image, mask_image=mask,
            height=size, width=size,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=1.0,                         # fully regenerate the masked region
            max_sequence_length=512,
            generator=gen,
        ).images[0]


# Qwen-Image enables true classifier-free guidance ONLY when a negative_prompt is
# present; with none, diffusers silently runs true_cfg_scale=1 (no guidance) and
# the --guidance knob is a dead no-op. Passing this engages true_cfg_scale (~4),
# so the trigger/texture prompt and LoRA actually steer — the representative way
# Qwen-Image is run, and the fix that stops it being under-guided vs FLUX/SDXL.
QWEN_NEGATIVE = ("blurry, low resolution, smooth flat fabric, plain cloth, "
                 "jpeg artifacts, washed out, painting, illustration, deformed")


class QwenInpaint:
    """
    Qwen-Image mask inpainting with a Qwen-Image LoRA.

    Defaults to QwenImageInpaintPipeline on the cached Qwen-Image base (no
    download) — mask-based, same (image, mask, prompt) call as the others. The
    higher-quality target is Qwen-Image-Edit-2509 via QwenImageEditInpaintPipeline
    (instruction + mask); that needs the ~20GB Edit-2509 download — switch the
    import + --base_id for it.

    Qwen uses true_cfg_scale for real CFG (guidance_scale is the distilled knob);
    we map --guidance onto true_cfg_scale and keep guidance_scale at 1.0.
    """
    base_id = "Qwen/Qwen-Image"
    default_trigger = "embfnchknt"
    default_size, default_steps, default_guidance = 1024, 30, 4.0

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.bfloat16,
                 base_id=None, transformer_file=None, components_id=None, fp8=False):
        from diffusers import QwenImageInpaintPipeline
        bid = base_id or self.base_id
        print(f"Loading {bid} …")
        self.pipe = QwenImageInpaintPipeline.from_pretrained(bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            # Split into dir + weight_name so it resolves under HF_HUB_OFFLINE=1
            # (offline can't auto-guess the weight filename from a bare path).
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if fp8:                                   # after LoRA — see helper docstring
            quantize_transformer_fp8(self.pipe.transformer)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            prompt=prompt, negative_prompt=QWEN_NEGATIVE,
            image=image, mask_image=mask,
            height=size, width=size,
            num_inference_steps=steps,
            true_cfg_scale=guidance,              # Qwen's real CFG knob (needs neg prompt)
            guidance_scale=1.0,
            strength=1.0,                         # fully regenerate the masked region
            max_sequence_length=512,
            generator=gen,
        ).images[0]


class QwenEditInpaint:
    """
    Qwen-Image-Edit-2511 instruction+mask inpainting with a Qwen-Image LoRA.

    The Edit variant is a fine-tune of Qwen-Image base that consumes the source
    image as an edit reference *and* an alpha mask, steered by an instruction
    prompt. Same concept LoRA as `qwen` (transformer layers are shared), but this
    path is instruction-driven — the right vehicle for the heterogeneous
    "switching" case ("replace the X region with embfnchknt french knots"), and
    generally higher quality than base mask-inpaint.

    Loads QwenImageEditInpaintPipeline from the cached Qwen-Image-Edit-2511 repo
    (scheduler/VAE/Qwen2.5-VL text-encoder/processor/transformer all present —
    no download). strength defaults to 0.6 in diffusers; we pass 1.0 to fully
    regenerate the masked region. true_cfg_scale is the real CFG knob.
    """
    base_id = "Qwen/Qwen-Image-Edit-2511"
    default_trigger = "embfnchknt"
    default_size, default_steps, default_guidance = 1024, 30, 4.0

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.bfloat16,
                 base_id=None, transformer_file=None, components_id=None, fp8=False):
        from diffusers import QwenImageEditInpaintPipeline
        if transformer_file:
            # Assemble from a local single-file transformer + the partial Edit-2511
            # cache, whose VAE/Qwen2.5-VL text-encoder/processor/scheduler/tokenizer
            # are all complete — only the transformer shards never downloaded. No fetch.
            from diffusers import QwenImageTransformer2DModel
            from transformers import Qwen2_5_VLForConditionalGeneration
            cid = components_id or self.base_id
            print(f"Loading Qwen-Edit transformer from single file: {transformer_file}")
            # QwenImageTransformer2DModel has no built-in single-file config mapping
            # (it falls back to the SD1.5 default → 404); point it at the repo's
            # transformer/config.json (a ~1KB fetch — only the multi-GB shards are slow).
            transformer = QwenImageTransformer2DModel.from_single_file(
                transformer_file, config=cid, subfolder="transformer", torch_dtype=dtype)
            # The cached Edit-2511 repo is missing 3 of 4 text_encoder shards (~12GB);
            # borrow the *identical* Qwen2.5-VL text encoder from the fully-cached
            # Qwen/Qwen-Image repo (same arch: hidden=3584, 28 layers). All other
            # components (vae/tokenizer/scheduler/processor) are complete in the cache.
            te_id = "Qwen/Qwen-Image"
            print(f"Borrowing text_encoder from cached: {te_id}")
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                te_id, subfolder="text_encoder", torch_dtype=dtype)
            print(f"Assembling QwenImageEditInpaintPipeline from cached components: {cid}")
            self.pipe = QwenImageEditInpaintPipeline.from_pretrained(
                cid, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype)
        else:
            bid = base_id or self.base_id
            print(f"Loading {bid} …")
            self.pipe = QwenImageEditInpaintPipeline.from_pretrained(bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            # Split into dir + weight_name so it resolves under HF_HUB_OFFLINE=1
            # (offline can't auto-guess the weight filename from a bare path).
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if fp8:                                   # after LoRA — see helper docstring
            quantize_transformer_fp8(self.pipe.transformer)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        # Qwen-Image-Edit consumes the source as a VISUAL REFERENCE (VL + VAE init),
        # so if the hole still shows the original stitches the model just copies them
        # back — near-identical output at every LoRA strength. Blank the masked region
        # to neutral grey in the reference so the model has nothing to reproduce and
        # must actually synthesise there. Only the *reference* is blanked; the mask
        # (white=regenerate) is unchanged, so the unmasked region still composites
        # from the true original.
        ref = image
        if mask is not None:
            a = np.asarray(image).copy()
            m = np.asarray(mask.resize(image.size, Image.NEAREST)) >= 128
            a[m] = 128                            # neutral grey hole -> forces generation
            ref = Image.fromarray(a)
        return self.pipe(
            image=ref, mask_image=mask, prompt=prompt, negative_prompt=QWEN_NEGATIVE,
            height=size, width=size,
            num_inference_steps=steps,
            true_cfg_scale=guidance,              # Qwen's real CFG knob (needs neg prompt)
            guidance_scale=1.0,
            strength=1.0,                         # fully regenerate the masked region
            max_sequence_length=512,
            generator=gen,
        ).images[0]


class QwenEditText:
    """
    Qwen-Image-Edit-2511 *maskless* pure-text editing with a Qwen-Image LoRA.

    No mask: the whole image is rewritten under a natural-language instruction
    that names the semantic target ("replace the hair with golden embfnchknt
    french knots"). This is the purest "switching" reintegration demo — only
    natural on whole-item scenes (hair/castle/snail), where a region can be
    referred to in words. Use --no_mask in main; the `mask` arg is ignored.

    QwenImageEditPipeline (not the inpaint variant) — same cached 2511 repo,
    no strength/mask args.
    """
    base_id = "Qwen/Qwen-Image-Edit-2511"
    default_trigger = "embfnchknt"
    default_size, default_steps, default_guidance = 1024, 30, 4.0

    def __init__(self, lora_path: str, cpu_offload: bool, dtype=torch.bfloat16,
                 base_id=None, transformer_file=None, components_id=None, fp8=False):
        from diffusers import QwenImageEditPipeline
        if transformer_file:
            from diffusers import QwenImageTransformer2DModel
            from transformers import Qwen2_5_VLForConditionalGeneration
            cid = components_id or self.base_id
            print(f"Loading Qwen-Edit transformer from single file: {transformer_file}")
            transformer = QwenImageTransformer2DModel.from_single_file(
                transformer_file, config=cid, subfolder="transformer", torch_dtype=dtype)
            # Edit-2511 cache is missing text_encoder shards; borrow the identical
            # Qwen2.5-VL text encoder from fully-cached Qwen/Qwen-Image (see QwenEditInpaint).
            te_id = "Qwen/Qwen-Image"
            print(f"Borrowing text_encoder from cached: {te_id}")
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                te_id, subfolder="text_encoder", torch_dtype=dtype)
            print(f"Assembling QwenImageEditPipeline from cached components: {cid}")
            self.pipe = QwenImageEditPipeline.from_pretrained(
                cid, transformer=transformer, text_encoder=text_encoder, torch_dtype=dtype)
        else:
            bid = base_id or self.base_id
            print(f"Loading {bid} (maskless edit) …")
            self.pipe = QwenImageEditPipeline.from_pretrained(bid, torch_dtype=dtype)
        if lora_path:
            print(f"Loading LoRA: {lora_path}")
            # Split into dir + weight_name so it resolves under HF_HUB_OFFLINE=1
            # (offline can't auto-guess the weight filename from a bare path).
            self.pipe.load_lora_weights(os.path.dirname(lora_path) or ".",
                                        weight_name=os.path.basename(lora_path),
                                        adapter_name="stitch")
        if fp8:                                   # after LoRA — see helper docstring
            quantize_transformer_fp8(self.pipe.transformer)
        if cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def set_strength(self, s: float):
        if s <= 0:
            self.pipe.disable_lora()
        else:
            self.pipe.enable_lora()
            self.pipe.set_adapters(["stitch"], [float(s)])

    def fill(self, image: Image.Image, mask: Image.Image, prompt: str,
             size: int, steps: int, guidance: float, seed: int) -> Image.Image:
        gen = torch.Generator("cpu").manual_seed(seed)
        return self.pipe(
            image=image, prompt=prompt,        # mask ignored — semantic targeting only
            negative_prompt=QWEN_NEGATIVE,
            height=size, width=size,
            num_inference_steps=steps,
            true_cfg_scale=guidance,           # needs neg prompt to engage CFG
            guidance_scale=1.0,
            max_sequence_length=512,
            generator=gen,
        ).images[0]


MODELS = {"flux1_fill": Flux1Fill, "flux1_inpaint": Flux1Inpaint, "sdxl": SdxlInpaint,
          "zimage": ZImageInpaint, "qwen": QwenInpaint,
          "qwen_edit": QwenEditInpaint, "qwen_edit_text": QwenEditText}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def round16(x: int) -> int:
    return max(16, (x // 16) * 16)


def prep(image_path: Path, mask_path: Path, size: int):
    """Load + square-resize image and mask to `size` (multiple of 16), aligned."""
    s = round16(size)
    img = Image.open(image_path).convert("RGB").resize((s, s), Image.LANCZOS)
    mask = Image.open(mask_path).convert("L").resize((s, s), Image.NEAREST)
    return img, mask


def make_strip(orig: Image.Image, mask: Image.Image, fill: Image.Image) -> Image.Image:
    """[ original | mask-overlay | fill ], or [ original | fill ] if mask is None."""
    o = np.asarray(orig).astype(np.float32)
    gap = np.full((o.shape[0], 4, 3), 200, np.uint8)
    if mask is None:                                  # maskless edit: no overlay panel
        strip = np.concatenate([np.asarray(orig), gap, np.asarray(fill)], axis=1)
        return Image.fromarray(strip)
    m = (np.asarray(mask) >= 128)[..., None]
    overlay = (o * (1 - 0.5 * m) + np.array([255, 60, 60]) * 0.5 * m).clip(0, 255).astype(np.uint8)
    strip = np.concatenate([np.asarray(orig), gap, overlay, gap, np.asarray(fill)], axis=1)
    return Image.fromarray(strip)


def fmt_strength(s: float) -> str:
    """0.5 -> '0.5', 1.0 -> '1.0' — stable token for texture_fidelity's --strength_regex."""
    return f"{s:.2f}".rstrip("0").rstrip(".") if s != int(s) else f"{int(s)}.0"


def png_meta(d: dict) -> PngInfo:
    """Embed generation params in PNG text chunks (like ComfyUI/A1111 do), so any
    fill is self-documenting and reproducible. Each key is its own tEXt entry, plus
    a combined 'parameters' line for quick eyeballing. Read back with PIL
    (Image.open(p).text) or `magick identify -verbose`."""
    info = PngInfo()
    for k, v in d.items():
        info.add_text(str(k), str(v))
    info.add_text("parameters", ", ".join(f"{k}: {v}" for k, v in d.items()))
    return info


def seed_from_surround(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Fill the masked hole with SURROUNDING colour only (cv2 Telea inpaint + heavy
    blur), so a sub-1.0 denoise seed carries a low-frequency colour anchor but NO
    ground-truth texture. This stops FLUX inventing a centred structure in a void
    without leaking the answer we're scoring against (unlike seeding with the
    original pixels). Outside the hole is untouched."""
    import cv2
    a = np.asarray(image.convert("RGB")).astype(np.uint8).copy()
    m = (np.asarray(mask.convert("L").resize(image.size, Image.NEAREST)) >= 128)
    m255 = (m.astype(np.uint8)) * 255
    filled = cv2.inpaint(a, m255, inpaintRadius=8, flags=cv2.INPAINT_TELEA)  # surround-only
    blur = cv2.GaussianBlur(filled, (0, 0), sigmaX=21)                       # kill fake structure
    out = np.where(m[..., None], blur, a)
    return Image.fromarray(out.astype(np.uint8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="flux1_fill")
    ap.add_argument("--base_id", default=None,
                    help="Override the model's base checkpoint (e.g. an inpaint variant)")
    ap.add_argument("--transformer_file", default=None,
                    help="FLUX: local single-file Fill transformer (.safetensors); "
                         "skips the repo download, borrows components from --components_id")
    ap.add_argument("--components_id", default=None,
                    help="FLUX: repo for VAE/T5/CLIP/scheduler (default FLUX.1-dev)")
    ap.add_argument("--images", required=True, help="Directory of holdout images")
    ap.add_argument("--mask_dir", default=None,
                    help="A single mask directory (white=regenerate); matched by stem")
    ap.add_argument("--mask_root", default=None,
                    help="Parent of per-size mask dirs; use with --mask_sizes")
    ap.add_argument("--mask_sizes", default=None,
                    help="Comma list of size subdirs under --mask_root, e.g. 128x128,256x256")
    ap.add_argument("--lora", required=True, help="LoRA .safetensors path")
    ap.add_argument("--trigger", default=None, help="Trigger token (default: model's)")
    ap.add_argument("--prompt", default=None,
                    help="Full prompt override; '' = empty prompt; else built from --trigger")
    ap.add_argument("--plain_prompt", default=None,
                    help="Plain-English texture DESCRIPTION for the strength-0 baseline "
                         "(no trigger) and the minimal-prompt models. Default: "
                         "'embroidery texture'. 'What someone without a LoRA would type'.")
    ap.add_argument("--desc", default=None,
                    help="FLUX-FILL ONLY: rich orb-suppressing texture description used at "
                         "BOTH strengths (s>0 = '{trigger}, {desc}', s0 = '{desc}'). The "
                         "keyword cannot replace these words for FLUX-Fill (it still orbs), "
                         "so Fill keeps the description; flagged in the write-up.")
    ap.add_argument("--denoise", type=float, default=None,
                    help="flux1_inpaint strength: 1.0 = fully regenerate the hole (no GT "
                         "leak); <1.0 (e.g. 0.8) keeps a faint seed so there's no void to "
                         "orb into. Ignored by models without a strength knob.")
    ap.add_argument("--seed_surround", action="store_true",
                    help="flux1_inpaint + denoise<1.0: seed the hole with blurred "
                         "SURROUNDING colour (cv2 Telea) instead of the original pixels, so "
                         "the seed carries no ground truth. REQUIRED for valid scored runs.")
    ap.add_argument("--label", default="",
                    help="Condition tag appended to filenames (e.g. kw, plain, none)")
    ap.add_argument("--strengths", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--size", type=int, default=None, help="Default: model-specific")
    ap.add_argument("--steps", type=int, default=None, help="Default: model-specific")
    ap.add_argument("--guidance", type=float, default=None, help="Default: model-specific")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N images (sorted by stem); 0 = all. "
                         "Used by the staged sweep to fill a fast subset first.")
    ap.add_argument("--no_cpu_offload", action="store_true",
                    help="Keep whole pipeline on GPU (needs ~24GB+ VRAM)")
    ap.add_argument("--fp8", action="store_true",
                    help="Quantize the transformer to fp8 (torchao). Needed for the "
                         "20B Qwen models on <=32GB cards: bf16 (~40GB) spills to "
                         "shared memory and crawls. Qwen models only.")
    ap.add_argument("--no_mask", action="store_true",
                    help="Maskless pure-text edit (qwen_edit_text): rewrite the whole "
                         "image under --prompt, no mask. For whole-item switching demos.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    strengths = [float(s) for s in args.strengths.split(",")]
    model_cls = MODELS[args.model]
    size = args.size or model_cls.default_size
    steps = args.steps or model_cls.default_steps
    guidance = args.guidance if args.guidance is not None else model_cls.default_guidance
    # fp8 only applies to the Qwen classes; pass it conditionally so Flux/SDXL/
    # Z-Image __init__ (which don't take fp8) are unaffected when the flag is off.
    extra = {"fp8": True} if args.fp8 else {}
    model = model_cls(args.lora, cpu_offload=not args.no_cpu_offload, base_id=args.base_id,
                      transformer_file=args.transformer_file, components_id=args.components_id,
                      **extra)
    if args.denoise is not None and hasattr(model, "denoise"):
        model.denoise = args.denoise                  # flux1_inpaint seed level
    if hasattr(model, "seed_surround"):
        model.seed_surround = args.seed_surround       # GT-free seed for sub-1.0 denoise
    trigger = args.trigger or model.default_trigger
    # Prompt structure. DEFAULT (every model except flux1_fill): the s=0 BASELINE
    # gets a plain-English texture DESCRIPTION (--plain_prompt) — "what someone
    # without a LoRA would type" — and the LoRA fills (s>0) SUBSTITUTE the trained
    # trigger for that description ("{trigger} texture"), demonstrating the token
    # encapsulates the sentence. flux1_fill (--desc) is the EXCEPTION: its keyword
    # cannot displace FLUX-Fill's centred-orb prior, so it keeps the rich
    # description at both strengths (s>0 adds the trigger). Flagged in the write-up.
    if args.prompt is not None:
        prompt_lora = prompt_base = args.prompt        # explicit override: all strengths
    elif args.desc:                                    # flux1_fill: rich description, both strengths
        prompt_lora = f"{trigger}, {args.desc}"        # s>0: trigger + description (LoRA flips orb->clean)
        prompt_base = args.desc                         # s=0: same description, no trigger (honest baseline)
    else:                                              # all other models: minimal, LoRA does the work
        prompt_lora = f"{trigger} texture"             # s>0: trigger replaces the description
        prompt_base = args.plain_prompt or "embroidery texture"   # s=0: plain description
    print(f"Prompt (s>0): {prompt_lora!r}")
    print(f"Prompt (s=0): {prompt_base!r}   label={args.label!r}")

    images = sorted(p for p in Path(args.images).iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.limit:                                  # staged sweep: fast first-N subset
        images = images[:args.limit]
    lbl = f"__{args.label}" if args.label else ""
    out_dir = Path(args.out)

    # Maskless pure-text path (qwen_edit_text): rewrite the whole image under the
    # natural-language --prompt, no mask. One pass, tag "nomask".
    if args.no_mask:
        mask_tag = "nomask"
        size_out = out_dir / mask_tag
        (size_out / "strips").mkdir(parents=True, exist_ok=True)
        print(f"\n[{mask_tag}]")
        s_img = round16(size)
        for img_path in images:
            img = Image.open(img_path).convert("RGB").resize((s_img, s_img), Image.LANCZOS)
            seed = args.seed + (zlib.crc32(img_path.stem.encode()) & 0xFFFF)
            for s in strengths:
                stem = f"{img_path.stem}__{mask_tag}__s{fmt_strength(s)}{lbl}"
                out_png = size_out / f"{stem}.png"
                if out_png.exists():                          # resume: already done
                    print(f"  skip {stem}")
                    continue
                model.set_strength(s)
                prompt = prompt_base if s <= 0 else prompt_lora   # plain-English baseline at s0
                fill = model.fill(img, None, prompt, size, steps, guidance, seed)
                meta = png_meta({"model": args.model, "prompt": prompt, "strength": fmt_strength(s),
                                 "guidance": guidance, "steps": steps, "size": size, "seed": seed,
                                 "denoise": getattr(model, "denoise", 1.0),
                                 "lora": os.path.basename(args.lora), "mask": "nomask"})
                fill.save(out_png, pnginfo=meta)
                make_strip(img, None, fill).save(size_out / "strips" / f"{stem}.png")
                print(f"  {stem}")
        print(f"\nDone -> {out_dir}")
        return

    # Resolve the set of mask directories (single, or per-size under a root).
    if args.mask_root and args.mask_sizes:
        mask_dirs = [Path(args.mask_root) / s for s in args.mask_sizes.split(",")]
    elif args.mask_dir:
        mask_dirs = [Path(args.mask_dir)]
    else:
        ap.error("provide either --mask_dir or (--mask_root and --mask_sizes)")

    for mask_dir in mask_dirs:
        mask_tag = mask_dir.name                              # e.g. "512x512"
        size_out = out_dir / mask_tag
        (size_out / "strips").mkdir(parents=True, exist_ok=True)
        print(f"\n[{mask_tag}]")

        for img_path in images:
            cand = list(mask_dir.glob(f"{img_path.stem}.*"))  # match a mask by stem
            if not cand:
                print(f"  no mask for {img_path.name}, skip")
                continue
            img, mask = prep(img_path, cand[0], size)
            seed = args.seed + (zlib.crc32(img_path.stem.encode()) & 0xFFFF)  # stable per image, shared across strengths

            for s in strengths:
                stem = f"{img_path.stem}__{mask_tag}__s{fmt_strength(s)}{lbl}"
                out_png = size_out / f"{stem}.png"
                if out_png.exists():                          # resume: already done
                    print(f"  skip {stem}")
                    continue
                model.set_strength(s)
                prompt = prompt_base if s <= 0 else prompt_lora   # plain-English baseline at s0
                fill = model.fill(img, mask, prompt, size, steps, guidance, seed)
                meta = png_meta({"model": args.model, "prompt": prompt, "strength": fmt_strength(s),
                                 "guidance": guidance, "steps": steps, "size": size, "seed": seed,
                                 "denoise": getattr(model, "denoise", 1.0),
                                 "lora": os.path.basename(args.lora), "mask": mask_tag})
                fill.save(out_png, pnginfo=meta)
                mask.save(size_out / f"{stem}.mask.png")      # for region-restricted scoring
                make_strip(img, mask, fill).save(size_out / "strips" / f"{stem}.png")
                print(f"  {stem}")

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()
