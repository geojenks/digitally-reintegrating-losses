# Sample data

For educational and academic use only.

## `stitches/`

20 sample training image+caption pairs per stitch type, drawn from the
full training sets (satin: 43 images, french knot and silk purl
similar scale). Each `<name>.png` has a descriptive `<name>.txt`
caption containing the trigger token (`embstn`, `embfnchknt`,
`embslkprl`). The `*_trigonly_captions/` folders hold the trigger-only
caption variants used to train the v2 FLUX adapters — same images,
captions reduced to the trigger token alone.

## `tiles_sample/`

Matched colour/normal RTI tile pairs (768×768), evenly sampled from
the 13,551-pair set used to fine-tune Marigold for surface-normal
estimation on embroidery. Same filename = same tile. 12 pairs live
here for browsing; a 100-pair sample (`tiles_sample_100_pairs.zip`)
is attached to the
[latest release](https://github.com/geojenks/digitally-reintegrating-losses/releases);
the full set and the fine-tuned checkpoint are on Zenodo:
https://doi.org/10.5281/zenodo.22828363

## `masks/`

The castle demonstration piece (`castle.png`, an embroidered casket
panel with simulated losses) and one binary mask per stitch type
(`castle__satin.png`, `castle__french_knot.png`,
`castle__silk_purl.png`) prescribing which stitch fills which loss.
