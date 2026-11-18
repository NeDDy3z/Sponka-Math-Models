# Sponka-Math-Models

Handwritten math to LaTeX models for [Sponka](https://github.com/NeDDy3z/Sponka), converted to ONNX so they run on the phone without internet.

In the app: Preferences > OCR > Handwritten math. Pick a model and download it once. The app reads the list of models from this repo's releases.

## Models

| Model | Tag | Size | Speed* | License | Good for |
|---|---|---|---|---|---|
| [TexTeller 3.0](https://github.com/NeDDy3z/Sponka-Math-Models/releases/tag/texteller-v3.0) (recommended) | `texteller-v3.0` | ~320 MB | ~6 s | Apache-2.0 | Long and complex formulas |
| [pix2text-mfr 1.5](https://github.com/NeDDy3z/Sponka-Math-Models/releases/tag/pix2text-mfr-v1.5) | `pix2text-mfr-v1.5` | ~120 MB | ~4 s | MIT | Light, fast, solid on handwriting |
| [PP-FormulaNet-L](https://github.com/NeDDy3z/Sponka-Math-Models/releases/tag/pp-formulanet-l-v1.0) | `pp-formulanet-l-v1.0` | ~370 MB | ~8 s | Apache-2.0 | Best accuracy, slowest |

*\*Speed was measured on a simple integral formula on an Exynos 1380 phone with 6 GB of RAM.*

From my limited testing, TexTeller 3.0 gives the most reliable results without any training in the app first.

## Making a release

1. Update `version` in `models/<id>/meta.json`, and the RAM and speed numbers if they changed.
2. Run the Release model workflow in Actions and pick the model.

The workflow tests the model on ONNX Runtime 1.30.0, the version the app uses, before it publishes.

Locally (Python 3.11 or 3.12):

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd export && python pix2text_mfr.py && cd ..
python check.py pix2text-mfr
python release.py pix2text-mfr --publish
```

PP-FormulaNet-L needs its own venv, see the top of `export/pp_formulanet_l.py`.

## Licenses

The scripts in this repo are under the [MIT license](LICENSE). Each model keeps its own license, listed above and in its release.
