"""
SAM-driven mask creator for inpaint sample preparation.

Setup:
    pip install gradio
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # or:
    # pip install segment-anything

    Place sam_vit_h_4b8939.pth in the same directory as this script.

Run:
    python sam_mask_tool.py
    Open http://127.0.0.1:7860 in your browser.

Workflow:
    1. Drop your source images (damaged textile photos, test images, etc.) into
       ./satin_inpaint_samples/sources/
    2. Refresh the dropdown, pick an image
    3. Click on the loss/missing area to add a positive point — SAM segments to that boundary.
       Add more positive points to extend the mask. Switch to "Negative" to clip false-positives.
    4. Click "Save .inpaint.png" — output goes to ./satin_inpaint_samples/<name>.inpaint.png,
       a 4-channel PNG with alpha=0 in the masked area (the format Flex2 wants).
"""
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# === Paths & setup ===
ROOT = Path(__file__).parent
CHECKPOINT_PATH = ROOT / "sam_vit_h_4b8939.pth"
SAMPLES_DIR = ROOT / "satin_inpaint_samples"
SOURCES_DIR = SAMPLES_DIR / "sources"
SOURCES_DIR.mkdir(parents=True, exist_ok=True)

if not CHECKPOINT_PATH.exists():
    raise SystemExit(
        f"SAM checkpoint not found at {CHECKPOINT_PATH}\n"
        f"Drop sam_vit_h_4b8939.pth into {ROOT} and re-run."
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM (vit_h) on {device}…")
sam = sam_model_registry["vit_h"](checkpoint=str(CHECKPOINT_PATH))
sam.to(device=device)
predictor = SamPredictor(sam)
print("SAM ready.")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}

# === Helpers ===
def list_sources():
    return sorted(str(p) for p in SOURCES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path):
    if not path:
        return None, None, [], []
    img = np.array(Image.open(path).convert("RGB"))
    predictor.set_image(img)
    return img, img, [], []  # display, image_state, pos, neg


def predict_mask(pos, neg):
    if not pos:
        return None
    points = np.array(pos + neg, dtype=np.float32)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=False,
    )
    return masks[0]  # bool (H, W)


def draw_points(img, points, color, radius=10):
    out = img.copy()
    H, W = out.shape[:2]
    for x, y in points:
        cx, cy = int(x), int(y)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < W and 0 <= py < H:
                        out[py, px] = color
    return out


def render(image, pos, neg):
    if image is None:
        return None
    overlay = image.copy()
    mask = predict_mask(pos, neg)
    if mask is not None:
        red = np.array([255, 0, 0], dtype=np.float32)
        overlay = overlay.astype(np.float32)
        overlay[mask] = overlay[mask] * 0.45 + red * 0.55
        overlay = overlay.astype(np.uint8)
    overlay = draw_points(overlay, pos, color=(0, 255, 0))     # green = include
    overlay = draw_points(overlay, neg, color=(255, 0, 255))   # magenta = exclude
    return overlay


def add_point(image, pos, neg, mode, evt: gr.SelectData):
    if image is None:
        return None, pos, neg
    x, y = evt.index[0], evt.index[1]
    if mode == "Positive (include)":
        pos = pos + [[x, y]]
    else:
        neg = neg + [[x, y]]
    return render(image, pos, neg), pos, neg


def reset_points(image):
    return render(image, [], []), [], []


def undo_point(image, pos, neg, mode):
    if mode == "Positive (include)" and pos:
        pos = pos[:-1]
    elif mode == "Negative (exclude)" and neg:
        neg = neg[:-1]
    return render(image, pos, neg), pos, neg


def save_inpaint(image, pos, neg, source_path):
    if image is None:
        return "No image loaded."
    if not pos:
        return "Add at least one positive point before saving."
    mask = predict_mask(pos, neg)
    rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = image
    rgba[..., 3] = (~mask).astype(np.uint8) * 255  # alpha 0 inside mask, 255 outside
    src = Path(source_path)
    out_path = SAMPLES_DIR / f"{src.stem}.inpaint.png"
    Image.fromarray(rgba, mode="RGBA").save(out_path)
    px = int(mask.sum())
    pct = 100.0 * px / mask.size
    return f"Saved {out_path.name} — masked {px:,} px ({pct:.1f}% of image)"


# === UI ===
with gr.Blocks(title="SAM mask creator") as demo:
    gr.Markdown(
        f"""### SAM mask creator for inpaint samples

1. Drop source images into `{SOURCES_DIR.relative_to(ROOT)}`
2. **Refresh** the list, pick an image
3. **Click on the loss area** (the part to be inpainted) — SAM segments to that boundary
4. Switch to **Negative** mode to clip over-segmented regions; **Positive** to extend
5. **Save** — writes `<name>.inpaint.png` to `{SAMPLES_DIR.relative_to(ROOT)}` (alpha=0 in masked area)

Output is in the format Flex2 expects. Green dots = include, magenta dots = exclude, red overlay = current mask."""
    )

    image_state = gr.State()  # original numpy
    pos_state = gr.State([])
    neg_state = gr.State([])

    with gr.Row():
        source_dd = gr.Dropdown(choices=list_sources(), label="Source image", scale=4)
        refresh_btn = gr.Button("Refresh list", scale=1)

    with gr.Row():
        display = gr.Image(type="numpy", interactive=True, label="Click to add point", height=720)
        with gr.Column(scale=1):
            mode = gr.Radio(
                ["Positive (include)", "Negative (exclude)"],
                value="Positive (include)",
                label="Click adds",
            )
            with gr.Row():
                undo_btn = gr.Button("Undo last point")
                reset_btn = gr.Button("Reset all points")
            save_btn = gr.Button("Save .inpaint.png", variant="primary")
            status = gr.Textbox(label="Status", lines=2, interactive=False)

    refresh_btn.click(lambda: gr.Dropdown(choices=list_sources()), outputs=source_dd)
    source_dd.change(
        load_image,
        inputs=source_dd,
        outputs=[display, image_state, pos_state, neg_state],
    )
    display.select(
        add_point,
        inputs=[image_state, pos_state, neg_state, mode],
        outputs=[display, pos_state, neg_state],
    )
    undo_btn.click(
        undo_point,
        inputs=[image_state, pos_state, neg_state, mode],
        outputs=[display, pos_state, neg_state],
    )
    reset_btn.click(
        reset_points,
        inputs=[image_state],
        outputs=[display, pos_state, neg_state],
    )
    save_btn.click(
        save_inpaint,
        inputs=[image_state, pos_state, neg_state, source_dd],
        outputs=status,
    )


if __name__ == "__main__":
    demo.launch()
