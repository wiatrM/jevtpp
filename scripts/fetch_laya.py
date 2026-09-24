#!/usr/bin/env python3
"""Download and verify the pinned multilingual Laya ONNX bundle."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import tempfile
import urllib.request


REPOSITORY = "codenamev/laya-onnx"
REVISION = "1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4"
FILES = {
    "model.onnx": "da6a0f87380597f679b12ce539e0a187e923fcffcf8dfc2f035728388dceacb0",
    "rl_agent_config.json": "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
    "onnx_config.json": "7eff4d0af9a8b22b977b690ab9e7d97ea2cda6c0a449e53502c06f7db2ae915f",
    "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    "tokenizer/tokenizer_config.json": "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1",
}
REMOTE_PREFIX = "multilingual"


def digest(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def fetch(relative: str, expected: str, destination: pathlib.Path, *,
          repository: str = REPOSITORY, revision: str = REVISION,
          remote_prefix: str = REMOTE_PREFIX) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and digest(destination) == expected:
        print(f"verified  {relative}")
        return
    remote = f"{remote_prefix}/{relative}"
    url = f"https://huggingface.co/{repository}/resolve/{revision}/{remote}?download=true"
    print(f"download  {relative}")
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = pathlib.Path(temporary.name)
            with urllib.request.urlopen(url) as response:
                shutil.copyfileobj(response, temporary, length=1024 * 1024)
        actual = digest(temporary_path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=pathlib.Path, help="target model directory")
    args = parser.parse_args()
    try:
        for relative, expected in FILES.items():
            fetch(relative, expected, args.directory / relative)
    except (OSError, RuntimeError) as error:
        print(f"fetch_laya: {error}", file=sys.stderr)
        return 1
    print(f"ready: {args.directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
