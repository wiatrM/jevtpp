#!/usr/bin/env python3
"""Download the pinned safetensors source of the multilingual ONNX export."""
import argparse
from pathlib import Path
import sys
from fetch_laya import fetch

REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
FILES = {
    "model.safetensors": "9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204",
    "encoder/config.json": "83f6916d13ef0f556ac461f28308dc2bffa7ebeadee8ec9e2db5812020ea5bb4",
    "rl_agent_config.json": "25061739243b617ad88d1219ba6f8a9c86c5881ca28df024fa2d9b3b2fcc30c6",
    "tokenizer/tokenizer.json": "609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f",
    "tokenizer/tokenizer_config.json": "6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1",
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        for relative, expected in FILES.items():
            fetch(relative, expected, args.directory / relative,
                  repository="convaiinnovations/laya", revision=REVISION,
                  remote_prefix="multilingual")
    except (OSError, RuntimeError) as error:
        print(f"fetch_laya_native: {error}", file=sys.stderr)
        return 1
    print(f"ready: {args.directory.resolve()} (revision {REVISION})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
