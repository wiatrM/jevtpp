"""Contract tests only: mocks never establish that CUDA inference has run."""
from types import SimpleNamespace
import unittest

from open_jev_runtime import inspect_cuda_runtime, install_runtime_endpoint, monitor_inference, require_cuda_device


def fake_torch(*, available=True, name="NVIDIA GeForce RTX 4090"):
    def device(value):
        kind, _, index = str(value).partition(":")
        return SimpleNamespace(type=kind, index=int(index) if index else None)
    return SimpleNamespace(
        device=device,
        cuda=SimpleNamespace(is_available=lambda: available, current_device=lambda: 0,
                             device_count=lambda: 1, get_device_name=lambda index: name,
                             memory_allocated=lambda device: 128, memory_reserved=lambda device: 256,
                             max_memory_allocated=lambda device: 512,
                             get_device_properties=lambda device: SimpleNamespace(total_memory=1024)),
        __version__="test-only", version=SimpleNamespace(cuda="test-only"))


def fake_predictor(*, parameter_device="cuda:0", buffer_device="cuda:0", input_device="cuda:0"):
    def tensor(device):
        return SimpleNamespace(device=device, dtype="torch.bfloat16", numel=lambda: 8)
    model = SimpleNamespace(
        named_parameters=lambda: iter([("backbone.weight", tensor(parameter_device)),
                                       ("head.weight", tensor("cuda:0"))]),
        named_buffers=lambda: iter([("backbone.position", tensor(buffer_device))]),
        device_name=input_device)
    return SimpleNamespace(scorer=SimpleNamespace(model=model), model_name="fixture-only",
                           method="fixture-only", provenance={"checkpoint_sha256": "fixture-only"})


class CudaRuntimeContract(unittest.TestCase):
    def test_cpu_request_refused(self):
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            require_cuda_device(fake_torch(), "cpu")

    def test_unavailable_cuda_refused(self):
        with self.assertRaisesRegex(RuntimeError, "fallback is disabled"):
            require_cuda_device(fake_torch(available=False), "cuda:0")

    def test_wrong_gpu_refused(self):
        with self.assertRaisesRegex(RuntimeError, "Expected GPU"):
            require_cuda_device(fake_torch(name="NVIDIA GeForce RTX 3090"), "cuda:0")

    def test_invalid_device_index_refused(self):
        with self.assertRaisesRegex(RuntimeError, "index 1 is unavailable"):
            require_cuda_device(fake_torch(), "cuda:1")

    def test_cpu_or_meta_weights_refused(self):
        for device in ("cpu", "meta", "cuda:1"):
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeError, "not fully"):
                inspect_cuda_runtime(fake_torch(), fake_predictor(parameter_device=device), "cuda:0")

    def test_cpu_buffers_refused(self):
        with self.assertRaisesRegex(RuntimeError, "not fully"):
            inspect_cuda_runtime(fake_torch(), fake_predictor(buffer_device="cpu"), "cuda:0")

    def test_input_device_mismatch_refused(self):
        with self.assertRaisesRegex(RuntimeError, "input device"):
            inspect_cuda_runtime(fake_torch(), fake_predictor(input_device="cpu"), "cuda:0")

    def test_runtime_reports_inspected_tensors(self):
        runtime = inspect_cuda_runtime(fake_torch(), fake_predictor(), "cuda")
        self.assertEqual(runtime["device"], "cuda:0")
        self.assertEqual(runtime["parameter_count"], 16)
        self.assertEqual(runtime["parameter_devices"], ["cuda:0"])
        self.assertEqual(runtime["buffer_devices"], ["cuda:0"])
        self.assertTrue(runtime["validated"])
        self.assertEqual(runtime["memory"]["process_allocated_bytes"], 128)

    def test_monitor_retains_actual_metadata_without_changing_answers(self):
        calls = []
        expected = {"model": "fixture-only", "answers": {"q0": {"choice": "option_1"}},
                    "usage": {"input_tokens": 123}, "metadata": {"candidate_sequences": 6}}
        predictor = fake_predictor()
        predictor.predict = lambda request: calls.append(request) or expected
        recent = monitor_inference(predictor)
        request = {"state": {"x": 1}, "questions": {"q0": {}}}
        self.assertIs(predictor.predict(request), expected)
        self.assertEqual(calls, [request])
        self.assertEqual(recent["last_inference"]["usage"], expected["usage"])
        self.assertEqual(recent["last_inference"]["sequence"], 1)
        self.assertEqual(len(recent["last_inference"]["state_sha256"]), 64)

    def test_monitor_does_not_mask_model_failure(self):
        predictor = fake_predictor()
        def fail(request):
            raise RuntimeError("model failed")
        predictor.predict = fail
        recent = monitor_inference(predictor)
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            predictor.predict({"state": {}, "questions": {"q0": {}}})
        self.assertIsNone(recent["last_inference"])

    def test_health_and_runtime_use_same_inspector_and_preserve_other_routes(self):
        class Upstream:
            def do_GET(self):
                self.upstream_called = True
        server = SimpleNamespace(RequestHandlerClass=Upstream)
        observed = []
        def inspect():
            observed.append(True)
            return {"validated": True, "device": "cuda:0"}
        install_runtime_endpoint(server, fake_predictor(), inspect)
        handler = server.RequestHandlerClass()
        responses = []
        handler.send = lambda status, data: responses.append((status, data))
        for path in ("/api/runtime", "/health?cache=1"):
            handler.path = path
            handler.do_GET()
        self.assertEqual(len(observed), 2)
        self.assertEqual([r[0] for r in responses], [200, 200])
        self.assertEqual(responses[0][1]["runtime"]["device"], "cuda:0")
        handler.path = "/v1/models"
        handler.do_GET()
        self.assertTrue(handler.upstream_called)

    def test_health_fails_closed_if_model_moves_to_cpu(self):
        class Upstream:
            pass
        server = SimpleNamespace(RequestHandlerClass=Upstream)
        def inspect():
            raise RuntimeError("weights moved to CPU")
        install_runtime_endpoint(server, fake_predictor(), inspect)
        handler = server.RequestHandlerClass()
        responses = []
        handler.send = lambda status, data: responses.append((status, data))
        handler.path = "/health"
        handler.do_GET()
        self.assertEqual(responses[0][0], 503)
        self.assertFalse(responses[0][1]["runtime"]["validated"])


if __name__ == "__main__":
    unittest.main()
