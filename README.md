# Sponka-Math-Models

Handwritten math to LaTeX models for [Sponka](https://github.com/NeDDy3z/Sponka), converted to ONNX so they run on the phone without internet.

In the app: Preferences > OCR > Handwritten math. Pick a model, download it once, and Convert to math works offline from then on. The app reads the list of models from this repo's releases.

## Models

| Model | Tag | Size | License | Good for |
|---|---|---|---|---|
| pix2text-mfr 1.5 (default) | `pix2text-mfr-v1.5` | ~120 MB | MIT | Light, fast, solid on handwriting |
| TexTeller 3.0 | `texteller-v3.0` | large | Apache-2.0 | Long and complex formulas |
| UniMERNet tiny | `unimernet-v1.0` | large | Apache-2.0 | Complex printed and handwritten formulas |
| PP-FormulaNet-L | `pp-formulanet-l-v1.0` | largest | Apache-2.0 | Best accuracy, slowest |

Each release page lists the model's size, recommended RAM and speed per formula. The app shows the same details before you download.

`onnxruntime-v1.30.0` holds the ONNX Runtime library per CPU type. The app downloads it once and every model uses it.

## Release format

Tag `<id>-v<version>`, title `<Name> <version>`. Each model release has:

- `encoder_model.onnx` + `decoder_model.onnx`, or `model.onnx` for end-to-end models
- `tokenizer.json`
- `model.json`, which tells the app what the model is and how to run it. The release text is generated from it.

```json
{
  "id": "pix2text-mfr", "name": "pix2text-mfr", "version": "1.5",
  "description": "...", "trainedOn": "...", "license": "MIT", "website": "https://...",
  "recommendedRamMb": 3072, "minRamMb": 2048, "avgSeconds": 2.0,
  "pipeline": "encoder-decoder",
  "tokenizer": "byte-bpe",
  "input": { "width": 384, "height": 384, "channels": 3, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "keepAspect": false },
  "decode": { "bos": 1, "eos": 2, "skipBelow": 5, "maxTokens": 256, "beam": 4 },
  "files": [ { "name": "encoder_model.onnx", "size": 87510770, "sha256": "..." } ]
}
```

- `pipeline`: `encoder-decoder` (the app runs the decoder and offers several readings) or `end-to-end` (one graph from image to tokens)
- `tokenizer`: `byte-bpe` or `metaspace`
- `keepAspect`: false squeezes the image to the input size, true fits it and pads with white
- `skipBelow`: token ids below this are special and get dropped

## Making a release
Needs Python 3.11 or 3.12 (onnxruntime has no wheels for newer versions yet).

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd export && python pix2text_mfr.py && cd ..   # or texteller.py, unimernet.py --checkpoint <dir>, pp_formulanet_l.py
python check.py pix2text-mfr                   # accuracy, speed, peak memory on samples/*.png + *.tex
python release.py pix2text-mfr                 # writes build/<id>/model.json and RELEASE.md
python release.py pix2text-mfr --publish       # creates the GitHub release
```

Before publishing, put real RAM and speed numbers in `models/<id>/meta.json`. Phones are roughly 3-5x slower than `check.py` on a desktop. Add `--int8` to an export to make the model about 4x smaller and faster at a small accuracy cost. Use it for the big models.

## Licenses

The scripts in this repo are under [your license]. Each model keeps its own license, listed above and in its release.
