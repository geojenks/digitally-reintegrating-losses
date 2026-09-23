"""
Minimal local caption editor for the satin LoRA dataset.

Run:
    pip install gradio
    python caption_editor.py

Then open the URL it prints (http://127.0.0.1:7860 by default).
Captions are saved on every navigation (Prev / Next / Go) and on Save.
"""
from pathlib import Path
import gradio as gr

DATASET_DIR = Path(__file__).parent / "data_stitches" / "silk_purl-1024"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def load_pairs(folder: Path):
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [(str(img), img.with_suffix(".txt")) for img in images]


PAIRS = load_pairs(DATASET_DIR)
TOTAL = len(PAIRS)
if TOTAL == 0:
    raise SystemExit(f"No images found in {DATASET_DIR}")


def read_caption(idx: int) -> str:
    _, txt_path = PAIRS[idx]
    return txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""


def write_caption(idx: int, content: str) -> None:
    _, txt_path = PAIRS[idx]
    txt_path.write_text(content, encoding="utf-8")


def view(idx: int):
    idx = max(0, min(idx, TOTAL - 1))
    img_path, txt_path = PAIRS[idx]
    label = f"**{idx + 1} / {TOTAL}** — `{txt_path.name}`"
    return img_path, read_caption(idx), idx, label


def save_and_step(current_idx: int, current_caption: str, delta: int):
    write_caption(current_idx, current_caption)
    return view(current_idx + delta)


def save_only(current_idx: int, current_caption: str):
    write_caption(current_idx, current_caption)
    return view(current_idx)


def jump(current_idx: int, current_caption: str, target):
    write_caption(current_idx, current_caption)
    return view(int(target) - 1)


with gr.Blocks(title="Satin caption editor") as demo:
    gr.Markdown(
        "### tile caption editor\n"
        "Captions auto-save when you click **Prev / Next / Go / Save**. "
        "Closing the browser without clicking one of those will lose your unsaved edit on the current tile."
    )
    idx_state = gr.State(0)
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(type="filepath", height=720, interactive=False)
        with gr.Column(scale=1):
            position = gr.Markdown()
            caption = gr.Textbox(label="Caption", lines=5, autofocus=True)
            with gr.Row():
                prev_btn = gr.Button("◀ Prev")
                save_btn = gr.Button("Save")
                next_btn = gr.Button("Next ▶", variant="primary")
            with gr.Row():
                jump_box = gr.Number(label=f"Jump (1..{TOTAL})", value=1, precision=0, scale=2)
                jump_btn = gr.Button("Go", scale=1)

    outputs = [image, caption, idx_state, position]
    prev_btn.click(lambda i, c: save_and_step(i, c, -1), [idx_state, caption], outputs)
    next_btn.click(lambda i, c: save_and_step(i, c, +1), [idx_state, caption], outputs)
    save_btn.click(save_only, [idx_state, caption], outputs)
    jump_btn.click(jump, [idx_state, caption, jump_box], outputs)
    caption.submit(lambda i, c: save_and_step(i, c, +1), [idx_state, caption], outputs)
    demo.load(lambda: view(0), outputs=outputs)


if __name__ == "__main__":
    demo.launch()
