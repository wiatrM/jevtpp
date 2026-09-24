"""Generate small deterministic ONNX contract fixtures (requires pip install onnx).

Run: python tests/laya_contract_fixtures.py /tmp/laya-contract-fixtures
Then: BUILD/tests/jevt_laya_integration_tests MODEL_DIR /tmp/laya-contract-fixtures
The existing model bundle supplies tokenizer/config files; these graphs replace
only its model. No model weights are downloaded or copied by this script.
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


INPUTS = {
    "input_ids": (TensorProto.INT64, ["batch", "tokens"]),
    "attention_mask": (TensorProto.INT64, ["batch", "tokens"]),
    "marker_pos": (TensorProto.INT64, ["batch", "markers"]),
    "marker_mask": (TensorProto.BOOL, ["batch", "markers"]),
    "qtype": (TensorProto.INT64, ["batch"]),
}


def emit(directory, name, inputs=None, shape=(2, 2), dtype=TensorProto.FLOAT, values=None):
    count = 1
    for dimension in shape:
        count *= dimension
    tensor = helper.make_tensor("constant", dtype, shape, values if values is not None else [0] * count)
    graph = helper.make_graph(
        [helper.make_node("Constant", [], ["logits"], value=tensor)],
        name,
        [helper.make_tensor_value_info(key, kind, dims)
         for key, (kind, dims) in (INPUTS if inputs is None else inputs).items()],
        [helper.make_tensor_value_info("logits", dtype, list(shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    onnx.save(model, directory / f"{name}.onnx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, (kind, dims) in INPUTS.items():
        emit(args.directory, f"{name}-dtype", inputs={**INPUTS, name: (TensorProto.FLOAT, dims)})
        emit(args.directory, f"{name}-rank", inputs={**INPUTS, name: (kind, [*dims, 1])})
    emit(args.directory, "logits-dtype", dtype=TensorProto.DOUBLE)
    emit(args.directory, "logits-rank", shape=(2, 2, 1))
    emit(args.directory, "logits-wide", shape=(2, 3))
    emit(args.directory, "logits-reshaped", shape=(1, 4))
    emit(args.directory, "logits-short", shape=(2, 1))
    for name, value in [("nan", float("nan")), ("inf", float("inf")), ("negative-inf", -float("inf"))]:
        emit(args.directory, f"logits-{name}", values=[0, 0, 0, value])
    emit(args.directory, "valid")
    emit(args.directory, "padded-negative-inf", values=[0, -float("inf"), 0, -float("inf")])


if __name__ == "__main__":
    main()
