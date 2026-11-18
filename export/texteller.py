"""TexTeller 3.0: a Hugging Face VisionEncoderDecoder, exported with optimum (no KV cache)."""
import argparse
import json
import pathlib
import subprocess

from huggingface_hub import hf_hub_download

from common import build_dir, quantize_int8, special_ids_below, write_export

REPO = "OleehyO/TexTeller"
# TexTeller's own preprocessing (texteller/utils/image.py): grey, fitted into 448 px, placed
# top-left, the rest padded with zero after normalizing (the mean grey).
SIZE = 448
MEAN = 0.9545467
STD = 0.15394445


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    out = build_dir("texteller")
    if not (out / "decoder_model.onnx").exists():
        subprocess.run(["optimum-cli", "export", "onnx", "--model", REPO, "--task", "image-to-text", str(out)], check=True)
    # The exported config drops the token ids, so they come from the original.
    config = json.loads(pathlib.Path(hf_hub_download(REPO, "config.json")).read_text())["decoder"]
    write_export(
        "texteller",
        input={"width": SIZE, "height": SIZE, "channels": 1, "mean": [MEAN], "std": [STD], "keepAspect": True, "padding": "top-left-mean"},
        decode={
            "bos": config["decoder_start_token_id"], "eos": config["eos_token_id"],
            "skipBelow": special_ids_below(out / "tokenizer.json"), "maxTokens": 512, "beam": 2,
        },
    )
    keep = {"encoder_model.onnx", "decoder_model.onnx", "tokenizer.json", "export.json"}
    for f in out.iterdir():
        if f.name not in keep:
            f.unlink()
    if args.int8:
        quantize_int8([out / "encoder_model.onnx", out / "decoder_model.onnx"])


if __name__ == "__main__":
    main()
