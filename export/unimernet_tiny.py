"""
UniMERNet tiny: a Swin-style encoder and mBART decoder from the UniMERNet package, exported by
hand to two graphs (image -> hidden states, ids + hidden states -> logits). The package pins an
old transformers, so run this from its own venv:
  uv venv --python 3.11 .venv-unimernet && uv pip install --python .venv-unimernet/bin/python torch torchvision unimernet onnx onnxruntime huggingface_hub "numpy<2" "pyarrow<16"
"""
import argparse
import pathlib
import shutil

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from common import build_dir, quantize_int8, special_ids_below, write_export

REPO = "wanderkid/unimernet_tiny"
# UniMERNet's eval preprocessing (FormulaImageEvalProcessor): ink cropped tight, fitted into
# 192x672, centered on black, grey normalized with these values.
HEIGHT, WIDTH = 192, 672
MEAN, STD = 0.7931, 0.1738


def load(ckpt: pathlib.Path):
    from unimernet.models.unimernet.unimernet import UniMERModel

    cfg = OmegaConf.create({
        "model_name": "unimernet",
        "model_config": {"model_name": str(ckpt), "max_seq_len": 1536},
        "tokenizer_name": "unimernet",
        "tokenizer_config": {"path": str(ckpt)},
        "load_pretrained": True,
        "pretrained": str(ckpt / "unimernet_tiny.pth"),
    })
    return UniMERModel.from_config(cfg).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    ckpt = pathlib.Path(snapshot_download(REPO))
    out = build_dir("unimernet")
    uni = load(ckpt)
    m = uni.model.model

    m.requires_grad_(False)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            h = self.m.encoder(pixel_values).last_hidden_state
            if self.m.config.encoder.hidden_size != self.m.config.decoder.hidden_size and self.m.config.decoder.cross_attention_hidden_size is None:
                h = self.m.enc_to_dec_proj(h)
            return h

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, input_ids, encoder_hidden_states):
            return self.m.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, return_dict=True).logits

    image = torch.zeros(1, 3, HEIGHT, WIDTH)
    with torch.no_grad():
        hidden = Encoder()(image)
        torch.onnx.export(
            Encoder(), (image,), str(out / "encoder_model.onnx"), opset_version=17, dynamo=False,
            input_names=["pixel_values"], output_names=["last_hidden_state"],
            dynamic_axes={"pixel_values": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        )
        # Traced with several tokens, so the causal mask follows the real length instead of being fixed to one.
        ids = torch.tensor([[uni.tokenizer.tokenizer.bos_token_id] + [100] * 7])
        torch.onnx.export(
            Decoder(), (ids, hidden), str(out / "decoder_model.onnx"), opset_version=17, dynamo=False,
            input_names=["input_ids", "encoder_hidden_states"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "steps"}, "encoder_hidden_states": {0: "batch"}, "logits": {0: "batch", 1: "steps"}},
        )
    shutil.copy(ckpt / "tokenizer.json", out / "tokenizer.json")
    tok = uni.tokenizer.tokenizer
    write_export(
        "unimernet",
        input={
            "width": WIDTH, "height": HEIGHT, "channels": 3, "mean": [MEAN] * 3, "std": [STD] * 3,
            "keepAspect": True, "padding": "center-black", "margin": 0,
        },
        decode={
            "bos": tok.bos_token_id, "eos": tok.eos_token_id,
            "skipBelow": special_ids_below(out / "tokenizer.json"), "maxTokens": 512, "beam": 2,
        },
    )
    if args.int8:
        quantize_int8([out / "encoder_model.onnx", out / "decoder_model.onnx"])


if __name__ == "__main__":
    main()
