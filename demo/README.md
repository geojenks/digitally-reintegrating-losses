# Per-region variant picker

`region_picker.html` is a zero-dependency page for composing a final
reintegration by eye. Run the pipeline with `--region_variants N` and
it writes a `variants/` folder next to the outputs containing
`base.png`, `regions.js` and N candidate fills per loss region. Copy
`region_picker.html` into that folder and serve it:

```bash
python -m http.server 8000 --directory <run>/variants
```

Click a region to cycle its variants (shift-click cycles backwards),
then press the download button to save the composite.

A live version with pre-generated variants for the castle piece — plus
an upload-your-own-image mode with box/brush/magic-wand mask tools —
is on the project page:
https://geojenks.github.io/digitally-reintegrating-losses/
