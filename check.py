"""
Run an exported model the way the app does on samples/*.png (each with a .tex holding what was
written) and report how many it reads right, seconds per formula and peak memory. Use the numbers
for avgSeconds and the RAM fields in models/<id>/meta.json (phones are roughly 3-5x slower).
"""
import argparse
import json
import re
import pathlib
import resource
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent


def same_math(a: str, b: str) -> bool:
    """Braces and spaces don't change what is written; the app's own check ignores them too."""
    def norm(s: str) -> str:
        # What the app's cleanup removes: display delimiters, font wrappers, spacing.
        s = re.sub(r"\\(\[|\]|mathbf|mathrm|mathit|boldsymbol|boldmath|textrm|textbf|bf|rm)|~", "", s)
        return "".join(c for c in s if not c.isspace() and c not in "{}")

    return norm(a) == norm(b)


def preprocess(path: pathlib.Path, spec: dict) -> np.ndarray:
    w, h, ch = spec["width"], spec["height"], spec["channels"]
    img = Image.open(path).convert("RGB")
    # The app renders only the ink with a white margin (24 px unless the model says otherwise), so crop samples the same way.
    margin = spec.get("margin", 24)
    ink = np.asarray(img.convert("L")) < 128
    if ink.any():
        ys, xs = np.nonzero(ink)
        img = img.crop((max(0, xs.min() - margin), max(0, ys.min() - margin), min(img.width, xs.max() + margin + 1), min(img.height, ys.max() + margin + 1)))
    top_left = spec.get("padding") == "top-left-mean"
    fitted = None
    if spec["keepAspect"]:
        fit = min(w / img.width, h / img.height)
        small = img.resize((max(1, int(img.width * fit)), max(1, int(img.height * fit))), Image.BILINEAR)
        fitted = small.size
        img = Image.new("RGB", (w, h), "black" if spec.get("padding") == "center-black" else "white")
        img.paste(small, (0, 0) if top_left else ((w - small.width) // 2, (h - small.height) // 2))
    else:
        img = img.resize((w, h), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    if ch == 1:
        a = (a @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    a = (a - np.array(spec["mean"], dtype=np.float32)) / np.array(spec["std"], dtype=np.float32)
    if top_left and fitted:
        # Outside the image the model saw zeros after normalizing, not white.
        mask = np.ones((h, w), dtype=bool)
        mask[: fitted[1], : fitted[0]] = False
        a[mask] = 0.0
    return a.transpose(2, 0, 1)[None].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    args = ap.parse_args()
    d = ROOT / "build" / args.model_id
    meta = json.loads((ROOT / "models" / args.model_id / "meta.json").read_text())
    exp = json.loads((d / "export.json").read_text())
    names = {"encoder": "encoder_model.onnx", "decoder": "decoder_model.onnx", "model": "model.onnx",
             "pixels": "pixel_values", "ids": "input_ids", "hidden": "encoder_hidden_states", **exp.get("onnx", {})}
    dec = exp["decode"]
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    if meta["pipeline"] == "end-to-end":
        whole = ort.InferenceSession(str(d / names["model"]))
    else:
        enc = ort.InferenceSession(str(d / names["encoder"]))
        decoder = ort.InferenceSession(str(d / names["decoder"]))

    samples = sorted(ROOT.glob("samples/*.png"))
    if not samples:
        raise SystemExit("put handwritten samples in samples/ (x.png + x.tex)")
    right, times = 0, []
    for png in samples:
        want = png.with_suffix(".tex").read_text().strip()
        image = preprocess(png, exp["input"])
        start = time.perf_counter()
        if meta["pipeline"] == "end-to-end":
            ids = [int(i) for i in whole.run(None, {names["pixels"]: image})[0].reshape(-1)]
        else:
            hidden = enc.run(None, {names["pixels"]: image})[0]
            ids = [dec["bos"]]
            while len(ids) < dec["maxTokens"]:
                logits = decoder.run(None, {names["ids"]: np.array([ids], dtype=np.int64), names["hidden"]: hidden})[0]
                nxt = int(logits[0, -1].argmax())
                if nxt == dec["eos"]:
                    break
                ids.append(nxt)
        times.append(time.perf_counter() - start)
        got = tok.decode([i for i in ids if i >= dec["skipBelow"]])
        ok = same_math(got, want)
        right += ok
        print(f"{'ok ' if ok else 'BAD'} {png.name}: {got}")
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\n{right}/{len(samples)} right, {sum(times) / len(times):.2f} s per formula, peak memory {peak_mb:.0f} MB")


if __name__ == "__main__":
    main()
