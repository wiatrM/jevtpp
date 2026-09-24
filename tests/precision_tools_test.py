#!/usr/bin/env python3
"""Precision-tool safety and quality-gate regression tests (no model required)."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "scripts"))
from laya_precision import build_tensors, comparison, gate_failures, quality
from laya_latency import CONTEXT, PREFIX, QUESTIONS, Fixture
from optimize_laya import materialize_fp32_weights, topological_sort, validate_destination


class PrecisionToolsTest(unittest.TestCase):
    def test_preparation_matches_existing_latency_fixture(self):
        import numpy as np
        from types import SimpleNamespace
        fixture = Fixture.__new__(Fixture)
        fixture.agent = dict(head_max_len=256, max_len=1024)
        fixture.special = dict(cls_token=1, sep_token=2, mask_token=3, pad_token=0)
        fixture.mask_text = "<mask>"
        fixture.tokenizer = SimpleNamespace(encode=lambda text, **kwargs:
                                           SimpleNamespace(ids=[ord(char) for char in text]))
        actual, sizes = build_tensors(fixture, CONTEXT, QUESTIONS, PREFIX)
        self.assertEqual(sizes, [4, 2, 3, 2])
        for name, expected in fixture.tensors(4).items():
            np.testing.assert_array_equal(actual[name], expected)

    def test_sort_restores_boundary_cast_order(self):
        import onnx
        from onnx import helper
        graph = helper.make_graph([
            helper.make_node("Identity", ["cast"], ["y"]),
            helper.make_node("Cast", ["x"], ["cast"], to=onnx.TensorProto.FLOAT16),
        ], "fixture", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [1])])
        model = helper.make_model(graph)
        topological_sort(model)
        self.assertEqual([node.op_type for node in model.graph.node], ["Cast", "Identity"])
        onnx.checker.check_model(model)

    def test_gates_accept_boundary_and_reject_probability_drift(self):
        metric = comparison([[.8, .2], [.1, .9]], [[.75, .25], [.1, .9]])
        quality_metrics = dict(category_accuracy=.75, urgent_accuracy=.5)
        self.assertEqual(metric["argmax_disagreements"], 0)
        self.assertEqual(gate_failures(metric, quality_metrics, quality_metrics,
                                      dict(max_probability_delta=.051, max_accuracy_drop=0)), [])
        self.assertEqual(len(gate_failures(metric, quality_metrics, quality_metrics,
                                          dict(max_probability_delta=.04))), 1)

    def test_argmax_and_label_gates_are_distinct(self):
        metric = comparison([[.51, .49]], [[.49, .51]])
        failures = gate_failures(metric, dict(category_accuracy=.5, urgent_accuracy=.25),
                                dict(category_accuracy=.75, urgent_accuracy=.5),
                                dict(max_argmax_disagreements=0, min_category_accuracy=.75,
                                     min_urgent_accuracy=.5, max_accuracy_drop=.1))
        self.assertEqual(len(failures), 5)

    def test_label_metrics_and_tie_policy(self):
        measured = quality([[[.7, .2, .05, .05], [.5, .5]]], [("state", 0, True)])
        self.assertEqual(measured["category_accuracy"], 1)
        self.assertEqual(measured["urgent_accuracy"], 1)
        self.assertEqual(measured["urgent_brier"], .25)

    def test_comparison_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            comparison([[.5, .5]], [[1.]])

    def test_output_cannot_overwrite_or_live_inside_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "model.onnx").touch()
            with self.assertRaises(ValueError):
                validate_destination(source, source)
            with self.assertRaises(ValueError):
                validate_destination(source, source / "fp16")
            with self.assertRaises(FileExistsError):
                validate_destination(source, Path(temporary))
            self.assertEqual(validate_destination(source, Path(temporary) / "new")[0], source)

    def test_fp16_weight_cast_is_folded_without_changing_other_nodes(self):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper
        weight = numpy_helper.from_array(np.array([[1.5]], dtype=np.float16), "w16")
        graph = helper.make_graph([
            helper.make_node("Cast", ["w16"], ["w32"], to=onnx.TensorProto.FLOAT),
            helper.make_node("MatMul", ["x", "w32"], ["y"]),
        ], "fixture", [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1])],
            [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 1])], [weight])
        model = helper.make_model(graph)
        self.assertEqual(materialize_fp32_weights(model), 1)
        self.assertEqual([node.op_type for node in model.graph.node], ["MatMul"])
        self.assertEqual([value.name for value in model.graph.initializer], ["w32"])
        self.assertEqual(numpy_helper.to_array(model.graph.initializer[0]).dtype, np.float32)
        self.assertEqual(float(numpy_helper.to_array(model.graph.initializer[0])[0, 0]), 1.5)
        onnx.checker.check_model(model)


if __name__ == "__main__":
    unittest.main()
