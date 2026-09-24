#!/usr/bin/env python3
"""Offline regression coverage for both pinned bundle downloaders."""

import contextlib
import hashlib
import io
import pathlib
import sys
import tempfile
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import fetch_laya
import fetch_laya_native


def sha256(content):
    return hashlib.sha256(content).hexdigest()


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = pathlib.Path(self.directory.name)
        # Every test prohibits real HTTP, including regressions in cache handling.
        self.http = self.enterContext(mock.patch.object(
            fetch_laya.urllib.request, "urlopen",
            side_effect=AssertionError("unexpected HTTP request in offline test")))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def response(self, content):
        self.http.side_effect = lambda _url: io.BytesIO(content)

    def test_onnx_default_uses_exact_pinned_url(self):
        content = b"small stand-in for ONNX bytes"
        destination = self.root / "model.onnx"
        self.response(content)
        fetch_laya.fetch("model.onnx", sha256(content), destination)
        self.http.assert_called_once_with(
            "https://huggingface.co/codenamev/laya-onnx/resolve/"
            "1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4/multilingual/model.onnx?download=true")
        self.assertEqual(destination.read_bytes(), content)
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_native_main_passes_pinned_repository_revision_and_prefix(self):
        content = b"small stand-in for safetensors bytes"
        self.response(content)
        with mock.patch.object(fetch_laya_native, "FILES", {"model.safetensors": sha256(content)}), \
                mock.patch.object(sys, "argv", ["fetch_laya_native", str(self.root)]):
            self.assertEqual(fetch_laya_native.main(), 0)
        self.http.assert_called_once_with(
            "https://huggingface.co/convaiinnovations/laya/resolve/"
            "1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual/model.safetensors?download=true")
        self.assertEqual((self.root / "model.safetensors").read_bytes(), content)

    def test_onnx_main_fetches_the_manifest_using_default_configuration(self):
        with mock.patch.object(fetch_laya, "fetch") as fetch, \
                mock.patch.object(sys, "argv", ["fetch_laya", str(self.root)]):
            self.assertEqual(fetch_laya.main(), 0)
        self.assertEqual(fetch.call_args_list, [
            mock.call(relative, expected, self.root / relative)
            for relative, expected in fetch_laya.FILES.items()])

    def test_native_manifest_uses_matching_head_and_tokenizer_hashes(self):
        self.assertEqual(fetch_laya_native.FILES["model.safetensors"],
                         "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204")
        self.assertEqual(fetch_laya_native.FILES["encoder/config.json"],
                         "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4")
        self.assertEqual(fetch_laya.FILES["model.onnx"],
                         "da6a0f87380597f679b12ce539e0a187e923fcffcf8dfc2f035728388dceacb0")
        for relative in ("rl_agent_config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"):
            self.assertEqual(fetch_laya_native.FILES[relative], fetch_laya.FILES[relative])
        for manifest in (fetch_laya.FILES, fetch_laya_native.FILES):
            for expected in manifest.values():
                self.assertRegex(expected, r"^[0-9a-f]{64}$")

    def test_valid_cache_hit_never_opens_network_or_temporary_file(self):
        destination = self.root / "cached.bin"
        destination.write_bytes(b"cached")
        with mock.patch.object(fetch_laya.tempfile, "NamedTemporaryFile") as temporary:
            fetch_laya.fetch("cached.bin", sha256(b"cached"), destination)
        self.http.assert_not_called()
        temporary.assert_not_called()
        self.assertEqual(destination.read_bytes(), b"cached")

    def test_correct_hash_replaces_stale_file_only_after_download(self):
        destination = self.root / "model.bin"
        destination.write_bytes(b"old model")
        content = b"verified replacement"

        def response(_url):
            self.assertEqual(destination.read_bytes(), b"old model")
            return io.BytesIO(content)

        self.http.side_effect = response
        fetch_laya.fetch("model.bin", sha256(content), destination)
        self.assertEqual(destination.read_bytes(), content)
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_hash_mismatch_preserves_existing_file_and_removes_temporary(self):
        destination = self.root / "model.bin"
        destination.write_bytes(b"existing model")
        self.response(b"corrupted download")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch for model.bin"):
            fetch_laya.fetch("model.bin", sha256(b"expected model"), destination)
        self.assertEqual(destination.read_bytes(), b"existing model")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_hash_mismatch_does_not_create_missing_target(self):
        destination = self.root / "new.bin"
        self.response(b"corrupted download")
        with self.assertRaises(RuntimeError):
            fetch_laya.fetch("new.bin", sha256(b"expected model"), destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_network_open_failure_removes_temporary_and_preserves_target(self):
        destination = self.root / "model.bin"
        destination.write_bytes(b"existing model")
        self.http.side_effect = urllib.error.URLError("offline")
        with self.assertRaises(urllib.error.URLError):
            fetch_laya.fetch("model.bin", sha256(b"expected model"), destination)
        self.assertEqual(destination.read_bytes(), b"existing model")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_interrupted_download_removes_partial_temporary(self):
        class InterruptedResponse(io.BytesIO):
            def read(self, size=-1):
                if self.tell():
                    raise OSError("connection lost after first chunk")
                return super().read(size)

        destination = self.root / "model.bin"
        destination.write_bytes(b"existing model")
        self.http.side_effect = lambda _url: InterruptedResponse(b"partial bytes")
        with self.assertRaisesRegex(OSError, "connection lost"):
            fetch_laya.fetch("model.bin", sha256(b"expected model"), destination)
        self.assertEqual(destination.read_bytes(), b"existing model")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_nested_destination_parent_is_created(self):
        destination = self.root / "tokenizer" / "tokenizer.json"
        self.response(b"{}")
        fetch_laya.fetch("tokenizer/tokenizer.json", sha256(b"{}"), destination)
        self.assertEqual(destination.read_bytes(), b"{}")
        self.assertEqual(list(destination.parent.iterdir()), [destination])

    def test_both_main_functions_report_download_errors(self):
        for module in (fetch_laya, fetch_laya_native):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module, "fetch", side_effect=OSError("offline")), \
                    mock.patch.object(sys, "argv", [module.__name__, str(self.root)]):
                self.assertEqual(module.main(), 1)


if __name__ == "__main__":
    unittest.main()
