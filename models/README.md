# Trained adapters

The LoRA weights are GitHub Release assets (too large to keep in the
repo): https://github.com/geojenks/digitally-reintegrating-losses/releases

| File | Base | Size | Trigger |
|---|---|---|---|
| `french_knot_stitch_sdxl_lora_v1.safetensors` | SDXL | 85 MB | `embfnchknt` |
| `satin_stitch_sdxl_lora_v1.safetensors` | SDXL | 85 MB | `embstn` |
| `silk_purl_sdxl_lora_v1.safetensors` | SDXL | 85 MB | `embslkprl` |
| `french_knot_stitch_flux1_lora_trigonly_v2.safetensors` | FLUX.1-dev | 172 MB | `embfnchknt` |
| `satin_stitch_flux1_lora_trigonly_v2.safetensors` | FLUX.1-dev | 172 MB | `embstn` |
| `silk_purl_flux1_lora_trigonly_v2.safetensors` | FLUX.1-dev | 172 MB | `embslkprl` |
| `french_knot_stitch_qwen_lora_v1.safetensors` | Qwen-Image | 295 MB | `embfnchknt` |
| `satin_stitch_qwen_lora_v1.safetensors` | Qwen-Image | 295 MB | `embstn` |
| `silk_purl_qwen_lora_v1.safetensors` | Qwen-Image | 295 MB | `embslkprl` |
| `french_knot_stitch_zimage_lora_v1.safetensors` | Z-Image base | 85 MB | `embfnchknt` |
| `satin_stitch_zimage_lora_v1.safetensors` | Z-Image base | 85 MB | `embstn` |
| `silk_purl_zimage_lora_v1.safetensors` | Z-Image base | 85 MB | `embslkprl` |

After downloading, place each file where the pipeline expects it:

```
output/<run_name>/<run_name>.safetensors
```

e.g. `output/satin_stitch_sdxl_lora_v1/satin_stitch_sdxl_lora_v1.safetensors`.

The `v1` family was trained with full descriptive captions; the
`trigonly_v2` FLUX family with trigger-only captions (stronger,
less prompt-sensitive — the variant used for the paper's final
deployment figures). Training configs for every run are in
`../training/configs/`.

The SDXL, FLUX and Qwen LoRAs all load through
`pipeline/staged_reintegrate.py` (`--model sdxl_base` / `flux_dev` /
`qwen_edit` etc.). The Qwen LoRAs were trained on Qwen-Image base and
transfer onto Qwen-Image-Edit-2509 at inference. The Z-Image LoRAs are
not wired into the pipeline script — they were trained on Z-Image base
and load onto Z-Image-Turbo for mask inpainting (diffusers
`ZImageInpaintPipeline`, or a ComfyUI workflow). See
`../training/TRAINING_TO_INFERENCE.txt` for the full training-to-
inference map.

The fine-tuned Marigold surface-normals checkpoint (~3.5 GB, used for
verification) is on Zenodo: https://doi.org/10.5281/zenodo.22828363
