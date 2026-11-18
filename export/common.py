"""Shared helpers for the export scripts: where builds go, export.json, int8 quantization."""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"


def build_dir(model_id: str) -> pathlib.Path:
    d = BUILD / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_export(model_id: str, *, input: dict, decode: dict, onnx: dict | None = None) -> None:
    """How the app runs the files: image size and normalization, token ids, ONNX tensor names."""
    data = {"input": input, "decode": decode}
    if onnx:
        data["onnx"] = onnx
    (build_dir(model_id) / "export.json").write_text(json.dumps(data, indent=2))


def special_ids_below(tokenizer_json: pathlib.Path) -> int:
    """One past the highest special token id, so the app can drop every special token by id."""
    t = json.loads(tokenizer_json.read_text())
    ids = [a["id"] for a in t.get("added_tokens", []) if a.get("special", True)]
    return max(ids) + 1 if ids else 0


def quantize_int8(paths: list[pathlib.Path]) -> None:
    """Dynamic int8 weights: about 4x smaller and faster on phone CPUs, slightly less accurate."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for p in paths:
        tmp = p.with_suffix(".int8.onnx")
        constants_to_initializers(p, p)
        quantize_dynamic(str(p), str(tmp), weight_type=QuantType.QInt8, extra_options={"EnableSubgraph": True})
        tmp.replace(p)


def constants_to_initializers(src, dst) -> None:
    """
    paddle2onnx stores weights as Constant nodes, which the quantizer skips; turn the big ones
    (here and in loop bodies) into initializers so quantize_int8 can find them.
    """
    import onnx

    def walk(g):
        keep = []
        for n in g.node:
            value = next((a.t for a in n.attribute if a.name == "value"), None) if n.op_type == "Constant" else None
            if value is not None and len(value.dims) >= 2 and value.data_type == onnx.TensorProto.FLOAT:
                value.name = n.output[0]
                g.initializer.append(value)
                continue
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    walk(a.g)
            keep.append(n)
        del g.node[:]
        g.node.extend(keep)

    m = onnx.load(str(src))
    walk(m.graph)
    onnx.save(m, str(dst), save_as_external_data=False)



def scalar_branch_outputs(path, names: list[str]) -> None:
    """
    paddle2onnx lets some If branches inside the decoding loop yield shape [1] for a loop variable
    that starts as a scalar. ONNX Runtime 1.32 lets that pass, 1.30 (the app's) stops; reshaping
    those branch outputs to scalars makes both agree. [names] are the branch outputs to fix, as
    ONNX Runtime 1.30 names them in its error.
    """
    import onnx
    from onnx import TensorProto, helper

    def branch_of(g, name):
        for n in g.node:
            for a in n.attribute:
                if a.type == onnx.AttributeProto.GRAPH:
                    if n.op_type == "If" and name in [o.name for o in a.g.output]:
                        return a.g
                    found = branch_of(a.g, name)
                    if found is not None:
                        return found
        return None

    m = onnx.load(str(path))
    for name in names:
        branch = branch_of(m.graph, name)
        shape = name + ".scalar_shape"
        branch.node.append(helper.make_node("Constant", [], [shape], value=helper.make_tensor(shape, TensorProto.INT64, [0], [])))
        branch.node.append(helper.make_node("Reshape", [name, shape], [name + ".scalar"]))
        for o in branch.output:
            if o.name == name:
                o.name = name + ".scalar"
    onnx.save(m, str(path))
