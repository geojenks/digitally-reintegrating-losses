"""
Pre-downloads only the diffusers-format components ai-toolkit needs for each
model. Skips monolithic safetensors checkpoints, README/LICENSE, and sample
images that ai-toolkit doesn't load.

Run with the ai-toolkit venv:
    .\ai-toolkit\venv\Scripts\python.exe predownload_models.py
"""
import os

# Fast Rust downloader + tight read timeout so stalls retry quickly.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from huggingface_hub import snapshot_download, hf_hub_download

# Standard diffusers subdirectories ai-toolkit reads from. Patterns that
# don't match anything in a given repo are silently skipped, so listing
# extras (e.g. text_encoder_3 for models that only have one) is harmless.
DIFFUSERS_COMPONENTS = [
    "transformer/*",      # Flux1, Flux2, Qwen
    "unet/*",             # SDXL
    "text_encoder/*",
    "text_encoder_2/*",
    "text_encoder_3/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "tokenizer_3/*",
    "vae/*",
    "scheduler/*",
    "model_index.json",
]

models = [
    ("black-forest-labs/FLUX.2-klein-base-4B", "flux2"),
    ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl"),
    ("Qwen/Qwen-Image", "qwen"),
]

for repo_id, label in models:
    print(f"\n{'='*60}")
    print(f"Downloading {label}: {repo_id}")
    print(f"{'='*60}")
    snapshot_download(repo_id=repo_id, allow_patterns=DIFFUSERS_COMPONENTS)
    print(f"Done: {label}")

# Qwen accuracy recovery adapter (single file from a different repo).
print(f"\n{'='*60}")
print("Downloading Qwen ARA (accuracy recovery adapter)")
print(f"{'='*60}")
hf_hub_download(
    repo_id="ostris/accuracy_recovery_adapters",
    filename="qwen_image_torchao_uint3.safetensors",
)
print("Done: Qwen ARA")

# === FLUX.2-Klein-specific files for ai-toolkit ===
# ai-toolkit's Klein loader does NOT use the diffusers transformer/ files. Instead it pulls:
#   1. The BFL single-file checkpoint  flux-2-klein-base-4b.safetensors  from the Klein repo
#   2. A separate Qwen3-4B repo for the text encoder
#   3. ae.safetensors from ai-toolkit/flux2_vae as the VAE fallback
# All three are required; without them ai-toolkit kicks off its own downloads at training time.
print(f"\n{'='*60}")
print("Downloading FLUX.2-Klein BFL transformer (flux-2-klein-base-4b.safetensors, ~7 GB)")
print(f"{'='*60}")
hf_hub_download(
    repo_id="black-forest-labs/FLUX.2-klein-base-4B",
    filename="flux-2-klein-base-4b.safetensors",
)
print("Done: Klein BFL transformer")

print(f"\n{'='*60}")
print("Downloading Qwen3-4B (text encoder for Klein, ~8 GB)")
print(f"{'='*60}")
snapshot_download(repo_id="Qwen/Qwen3-4B")
print("Done: Qwen3-4B")

print(f"\n{'='*60}")
print("Downloading ai-toolkit/flux2_vae (ae.safetensors)")
print(f"{'='*60}")
hf_hub_download(repo_id="ai-toolkit/flux2_vae", filename="ae.safetensors")
print("Done: flux2 VAE")

print("\nAll downloads complete.")
