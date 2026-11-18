"""
PP-FormulaNet-L: fetched through PaddleX and converted with paddle2onnx. Its decoding loop is part
of the graph, so the result is one end-to-end model (image in, token ids out) and the app does no
beam search for it. Run from its own venv:
  uv venv --python 3.11 .venv-paddle && uv pip install --python .venv-paddle/bin/python paddlepaddle "paddlex[ocr]" paddle2onnx onnx onnxruntime pyyaml
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import sys

import onnx
import yaml

from common import build_dir, quantize_int8, scalar_branch_outputs, special_ids_below, write_export

# PaddleX's preprocessing (UniMERNetImgDecode, UniMERNetTestTransform, LatexImageFormat): ink
# cropped tight, fitted into 768x768, centered on black, grey normalized with these values.
SIZE = 768
MEAN, STD = 0.7931, 0.1738


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    out = build_dir("pp-formulanet-l")
    src = pathlib.Path.home() / ".paddlex/official_models/PP-FormulaNet-L"
    if not src.exists():
        from paddlex import create_model

        create_model("PP-FormulaNet-L")
    tmp = out / "onnx"
    paddlex = pathlib.Path(sys.executable).parent / "paddlex"
    subprocess.run(
        [str(paddlex), "--paddle2onnx", "--paddle_model_dir", str(src), "--onnx_model_dir", str(tmp), "--opset_version", "17"],
        check=True,
    )
    (tmp / "inference.onnx").replace(out / "model.onnx")
    shutil.rmtree(tmp)

    # The tokenizer travels inside inference.yml.
    chars = yaml.safe_load((src / "inference.yml").read_text())["PostProcess"]["character_dict"]
    (out / "tokenizer.json").write_text(json.dumps(chars["fast_tokenizer_file"]))
    config = chars["tokenizer_config_file"]
    vocab = json.loads((out / "tokenizer.json").read_text())["model"]["vocab"]
    graph = onnx.load(str(out / "model.onnx"), load_external_data=False).graph
    write_export(
        "pp-formulanet-l",
        input={
            "width": SIZE, "height": SIZE, "channels": 1, "mean": [MEAN], "std": [STD],
            "keepAspect": True, "padding": "center-black", "margin": 0,
        },
        decode={
            "bos": vocab[config["bos_token"]], "eos": vocab[config["eos_token"]],
            "skipBelow": special_ids_below(out / "tokenizer.json"), "maxTokens": 1024, "beam": 1,
        },
        onnx={"model": "model.onnx", "pixels": graph.input[0].name},
    )
    if args.int8:
        quantize_int8([out / "model.onnx"])
    # Found by running the model on ONNX Runtime 1.30, the version the app ships.
    scalar_branch_outputs(out / "model.onnx", ["p2o.yield.17"])


if __name__ == "__main__":
    main()
