"""
Turn build/<id>/ into a release: model.json (models/<id>/meta.json + build/<id>/export.json +
each file's size and sha256) and a release body made from the same data. With --publish it runs
gh release create <id>-v<version>. The app reads model.json, so the page and the app always agree.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
REPO = "NeDDy3z/Sponka-Math-Models"


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--publish", action="store_true", help="create the GitHub release (otherwise only write the files)")
    ap.add_argument("--prerelease", action="store_true", help="mark the release as a pre-release")
    args = ap.parse_args()
    d = ROOT / "build" / args.model_id
    meta = json.loads((ROOT / "models" / args.model_id / "meta.json").read_text())
    exp = json.loads((d / "export.json").read_text())
    names = ["model.onnx"] if meta["pipeline"] == "end-to-end" else ["encoder_model.onnx", "decoder_model.onnx"]
    names.append("tokenizer.json")
    files = [{"name": n, "size": (d / n).stat().st_size, "sha256": sha256(d / n)} for n in names]
    model = {**meta, **exp, "files": files}
    (d / "model.json").write_text(json.dumps(model, indent=2))

    mb = lambda b: round(b / 1_000_000)
    size = sum(f["size"] for f in files)
    body = "\n".join([
        model["description"],
        "",
        f"- Trained on: {model['trainedOn']}",
        f"- License: {model['license']}",
        f"- Website: {model['website']}",
        f"- Size: {mb(size)} MB",
        f"- Recommended RAM: {model['recommendedRamMb'] / 1024:.1f} GB (at least {model['minRamMb'] / 1024:.1f} GB)",
        f"- About {model['avgSeconds']:g} s per formula (measured on an Exynos 1380 phone with 6 GB of RAM)",
        "",
        "Downloaded and run by Sponka (Preferences > OCR > Handwritten math). model.json describes the files for the app.",
    ])
    (d / "RELEASE.md").write_text(body + "\n")
    tag = f"{model['id']}-v{model['version']}"
    print(f"wrote {d / 'model.json'} and RELEASE.md for {tag}")
    if args.publish:
        cmd = ["gh", "release", "create", tag, "-R", REPO, "--title", f"{model['name']} {model['version']}",
               "--notes-file", str(d / "RELEASE.md"), *[str(d / n) for n in names], str(d / "model.json")]
        if args.prerelease:
            cmd.append("--prerelease")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
