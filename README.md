# Sponka-Math-Models

Handwritten math to LaTeX models for [Sponka](https://github.com/NeDDy3z/Sponka), converted to ONNX so they run on the phone without internet.

In the app: Preferences > OCR > Handwritten math. Pick a model and download it once. The app reads the list of models from this repo's releases.

## Models

| Model | Tag | Size | Speed* | License | Good for |
|---|---|---|---|---|---|
| TexTeller 3.0 (recommended) | `texteller-v3.0` | ~320 MB | ~6 s | Apache-2.0 | Long and complex formulas |
| pix2text-mfr 1.5 | `pix2text-mfr-v1.5` | ~120 MB | ~4 s | MIT | Light, fast, solid on handwriting |
| PP-FormulaNet-L | `pp-formulanet-l-v1.0` | ~370 MB | ~8 s | Apache-2.0 | Best accuracy, slowest |

*\*Speed was measured on a simple integral formula on an Exynos 1380 phone with 6 GB of RAM.*

From my limited testing, TexTeller 3.0 gives the most reliable results without any training in the app first.

## Making a release

Bump `version` in `models/<id>/meta.json`, then run the **Release model** workflow in the Actions tab and pick the model. It exports the model, checks it on ONNX Runtime 1.30.0 (the app's version) and creates the release.

To do it locally instead (Python 3.11 or 3.12):

```sh
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd export && python pix2text_mfr.py && cd ..   # pp_formulanet_l.py needs its own venv, see the top of the script
python check.py pix2text-mfr                   # accuracy and speed on samples/
python release.py pix2text-mfr --publish       # writes model.json and creates the GitHub release
```

Put real RAM and speed numbers in `models/<id>/meta.json` before publishing. `model.json` in each release tells the app how to run the model.

## Licenses

The scripts in this repo are under [your license]. Each model keeps its own license, listed above and in its release.
