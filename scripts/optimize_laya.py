#!/usr/bin/env python3
"""Create an opt-in Laya FP16 or dynamic INT8 bundle without changing its source."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_destination(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("output must be separate from, and outside, the source bundle")
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}; choose a new directory")
    if not (source / "model.onnx").is_file():
        raise FileNotFoundError(source / "model.onnx")
    return source, destination


def materialize_fp32_weights(model):
    """Fold only Cast(initializer, FLOAT), as used by the pinned FP16-on-disk export.

    QuantizeDynamic recognizes constant matrix weights; leaving these Casts in
    place would silently skip them. Other Casts and graph computations remain.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper
    initializers = {value.name: value for value in model.graph.initializer}
    folded = []
    for node in model.graph.node:
        if (node.op_type == "Cast" and len(node.input) == 1 and len(node.output) == 1
                and node.input[0] in initializers
                and any(attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT
                        for attribute in node.attribute)):
            value = numpy_helper.to_array(initializers[node.input[0]]).astype(np.float32)
            model.graph.initializer.append(numpy_helper.from_array(value, node.output[0]))
            folded.append(node)
    for node in folded:
        model.graph.node.remove(node)
    used = {name for node in model.graph.node for name in node.input}
    used.update(value.name for value in model.graph.output)
    used.update(value.name for value in model.graph.input)
    kept = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return len(folded)


def topological_sort(model):
    """Converters append boundary Cast nodes; restore dependency order."""
    ready = {item.name for item in model.graph.input}
    ready.update(item.name for item in model.graph.initializer)
    pending, ordered = list(model.graph.node), []
    while pending:
        remaining = []
        for node in pending:
            if all(not name or name in ready for name in node.input):
                ordered.append(node)
                ready.update(node.output)
            else:
                remaining.append(node)
        if len(remaining) == len(pending):
            raise ValueError("converted graph has a cycle or unresolved input")
        pending = remaining
    del model.graph.node[:]
    model.graph.node.extend(ordered)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="original bundle with model.onnx")
    parser.add_argument("output", type=Path, help="new bundle directory; must not exist")
    parser.add_argument("--precision", required=True, choices=["fp16", "int8"])
    parser.add_argument("--block-op", action="append", default=[],
                        help="additional operation type to keep FP32 during FP16 conversion")
    parser.add_argument("--block-node", action="append", default=[],
                        help="node name to keep FP32 during FP16 conversion")
    parser.add_argument("--per-channel", action="store_true", help="INT8 per-channel weights")
    parser.add_argument("--disable-shape-infer", action="store_true",
                        help="FP16: use existing graph types if converter shape inference fails")
    args = parser.parse_args()
    if args.precision == "int8" and (args.block_op or args.block_node or args.disable_shape_infer):
        parser.error("block-op, block-node and disable-shape-infer apply only to FP16")
    if args.precision == "fp16" and args.per_channel:
        parser.error("per-channel applies only to INT8")
    try:
        source, destination = validate_destination(args.source, args.output)
        import onnx
        import onnxruntime as ort
        destination.mkdir(parents=True, exist_ok=False)
    except Exception as error:
        parser.exit(1, f"optimize_laya: {error}\n")
    started = time.perf_counter()
    report = dict(status="failed", source=str(source), precision=args.precision,
                  source_sha256=sha256(source / "model.onnx"), onnx=onnx.__version__,
                  onnxruntime=ort.__version__, keep_io_types=True,
                  block_ops=args.block_op, block_nodes=args.block_node,
                  per_channel=args.per_channel, disable_shape_infer=args.disable_shape_infer,
                  default_backend_changed=False)
    try:
        model = onnx.load(str(source / "model.onnx"))
        original_io = [(value.name, value.type.tensor_type.elem_type)
                       for value in [*model.graph.input, *model.graph.output]]
        output_model = destination / "model.onnx"
        if args.precision == "fp16":
            from onnxruntime.transformers import float16
            report["fp16_converter"] = "onnxruntime.transformers.float16"
            model = float16.convert_float_to_float16(
                model, keep_io_types=True,
                disable_shape_infer=args.disable_shape_infer,
                op_block_list=list(set(float16.DEFAULT_OP_BLOCK_LIST + args.block_op)),
                node_block_list=args.block_node)
            topological_sort(model)
            onnx.save_model(model, str(output_model))
        else:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            report["folded_weight_casts"] = materialize_fp32_weights(model)
            # Exported intermediate annotations can be stale after constant
            # folding. Let the quantizer recompute them; preserve public I/O.
            report["rebuilt_value_info_count"] = len(model.graph.value_info)
            del model.graph.value_info[:]
            quantize_dynamic(model, str(output_model), weight_type=QuantType.QInt8,
                             per_channel=args.per_channel, op_types_to_quantize=["MatMul", "Gemm"],
                             use_external_data_format=True,
                             extra_options={"MatMulConstBOnly": True})
        del model
        # Read just the graph to avoid a second copy of large external weights.
        converted = onnx.load(str(output_model), load_external_data=False)
        converted_io = [(value.name, value.type.tensor_type.elem_type)
                        for value in [*converted.graph.input, *converted.graph.output]]
        if original_io != converted_io:
            raise RuntimeError("conversion changed public input/output types or names")
        onnx.checker.check_model(str(output_model))
        op_counts = Counter(node.op_type for node in converted.graph.node)
        if args.precision == "int8" and not op_counts["MatMulInteger"]:
            raise RuntimeError("INT8 conversion produced no MatMulInteger nodes")
        report["operation_counts"] = dict(op_counts)
        report["initializer_types"] = dict(Counter(onnx.TensorProto.DataType.Name(value.data_type)
                                                   for value in converted.graph.initializer))
        for relative in ("rl_agent_config.json", "onnx_config.json", "laya_config.json"):
            if (source / relative).is_file():
                shutil.copy2(source / relative, destination / relative)
        shutil.copytree(source / "tokenizer", destination / "tokenizer")
        report.update(status="converted_unvalidated", output_sha256=sha256(output_model),
                      model_files_bytes=sum(path.stat().st_size for path in destination.glob("model.onnx*")),
                      artifact_sha256={path.name: sha256(path) for path in destination.glob("model.onnx*")})
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report["elapsed_seconds"] = time.perf_counter() - started
    (destination / "precision-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0 if report["status"] == "converted_unvalidated" else 1


if __name__ == "__main__":
    sys.exit(main())
