"""pix2text-mfr 1.5: the ONNX files are published by the authors, so this only fetches them."""
import argparse
import json
import shutil

from huggingface_hub import hf_hub_download

from common import build_dir, quantize_int8, special_ids_below, write_export

REPO = "breezedeus/pix2text-mfr-1.5"
REVISION = "1cef9f0"
FILES = ["encoder_model.onnx", "decoder_model.onnx", "tokenizer.json", "config.json", "preprocessor_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    out = build_dir("pix2text-mfr")
    for f in FILES:
        shutil.copy(hf_hub_download(REPO, f, revision=REVISION), out / f)
    config = json.loads((out / "config.json").read_text())
    pre = json.loads((out / "preprocessor_config.json").read_text())
    size = pre.get("size", {"height": 384, "width": 384})
    write_export(
        "pix2text-mfr",
        input={
            "width": size["width"], "height": size["height"], "channels": 3,
            "mean": pre["image_mean"], "std": pre["image_std"], "keepAspect": False,
        },
        decode={
            "bos": config["decoder_start_token_id"], "eos": config["eos_token_id"],
            "skipBelow": special_ids_below(out / "tokenizer.json"), "maxTokens": 256, "beam": 4,
        },
    )
    for f in ["config.json", "preprocessor_config.json"]:
        (out / f).unlink()
    if args.int8:
        quantize_int8([out / "encoder_model.onnx", out / "decoder_model.onnx"])


if __name__ == "__main__":
    main()
