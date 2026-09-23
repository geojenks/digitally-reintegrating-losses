# Job bundles

A job bundle is everything needed to reintegrate one image: the image, one
mask per layer, and a `job.json` saying which stitch fills each layer and how
its procedural init looks. The mask tool on the project page exports bundles;
`staged_reintegrate.py --job` and `run_jobs.py` run them, locally or in Colab.

## Layout

A bundle is a `.zip` (or an unzipped folder) with this at its root:

```
job.json
image.png            the image, any aspect ratio
masks/<layer id>.png one 8-bit greyscale mask per layer, same size as image.png,
                     white (>=128) = pixels this layer fills
loras/<file>         optional: custom LoRA weights named in job.json
```

## `job.json`

```json
{
  "format": "reintegration-job",
  "version": 1,
  "name": "castle",
  "image": "image.png",
  "layers": [
    {"id": "satin_1", "stitch": "satin", "mask": "masks/satin_1.png",
     "colour": "#eee8d6", "angle": 0, "thread": 0},
    {"id": "satin_lines", "stitch": "satin", "mask": "masks/satin_lines.png",
     "colour": "#1e1a18", "angle": 90},
    {"id": "knots", "stitch": "french_knot", "mask": "masks/knots.png",
     "colour": "#c87878", "knot_r": 16},
    {"id": "chain_1", "stitch": "chain_stitch", "mask": "masks/chain_1.png",
     "colour": "#3a5a8c"}
  ],
  "stitches": {
    "chain_stitch": {"lora": "loras/chain_stitch.safetensors", "trigger": "embchn",
                     "init": "satin", "prompt": "{trig}, {colour} {trig}"}
  },
  "order": ["french_knot", "silk_purl", "satin", "chain_stitch"],
  "settings": {"per_region": true, "brim": 10, "sib_feather": 3,
               "region_max_up": 2, "denoise": 0.6, "denoise_french_knot": 0.65,
               "denoise_satin": 0.5, "denoise_silk_purl": 0.8,
               "region_variants": 0, "seed": 1600, "size": 1024}
}
```

Fields:

- `name`: output folder name (`[A-Za-z0-9_-]+`).
- `layers[]`, in paint order. Where masks overlap, the later layer wins.
  - `id`: `[a-z0-9_]+`, unique.
  - `stitch`: `french_knot`, `satin`, `silk_purl`, or a key of `stitches`.
  - `mask`: path inside the bundle.
  - `colour`: `#rrggbb`, the base colour of this layer's procedural init.
  - Optional init shape: `angle` (satin thread angle in degrees, 0 = vertical
    threads), `thread` (satin thread width), `knot_r` (french knot radius),
    `coil` (silk purl coil pitch). Sizes are in pixels of the working image,
    whose long side is `size` (1024 by default, the same as the mask tool's
    working size). 0 or missing = the pipeline default.
  - Optional `prompt` (template with `{trig}` and `{colour}`) and `denoise`.
- `stitches` (optional): custom LoRAs. `lora` is relative to the bundle root or
  absolute; `trigger` is the trained token; `init` is the procedural init style
  (`satin`, `french_knot`, `silk_purl` or `flat`); `prompt` is optional.
  The LoRA must suit the model it runs on (a FLUX LoRA for `--model flux_base`).
- `order` (optional): stage order. Stitches not listed run afterwards in the
  order they first appear in `layers`. Default: french_knot, silk_purl, satin,
  then custom stitches.
- `settings` (optional): defaults for any `staged_reintegrate.py` flag, by its
  name without the dashes. Flags given on the command line win. The model is
  not set here: the runner chooses it.

## How a job runs

- One stage per stitch, so each LoRA is applied once.
- Within a stage, every layer paints its own procedural init (its colour,
  angle and so on) into its own mask. One diffusion pass then restyles the
  union of that stitch's layers. A coloured satin patch and black satin detail
  lines therefore share one pass, and the lines start out black.
- Other stitches' layers are protected as usual (the brim and sibling feather
  treat them as siblings). Layers of the same stitch are not siblings of each
  other.
- With `per_region`, each connected region of the stage's union is filled on
  its own. Its prompt and denoise come from the layer covering most of it.
- `{colour}` in a prompt becomes the nearest plain colour name to the layer's
  colour (for example "dark red", "cream", "black").
- Non-square images are padded to a square for the model and cropped back.
