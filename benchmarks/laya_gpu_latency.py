#!/usr/bin/env python3
"""Experimental CUDA latency probe for the same fixture as laya_latency.py.

Requires onnxruntime-gpu, CUDA/cuDNN libraries, onnx and tokenizers. This
measures Python + GPU headroom, not the C++ adapter (which currently uses CPU).
Tokenization, host/device copies and probability conversion are included.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import tempfile
import time

import onnx
import onnxruntime as ort

from laya_latency import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or args.threads < 1:
        parser.error('iterations and threads must be positive; warmup nonnegative')
    ort.preload_dlls()
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    providers = [('CUDAExecutionProvider', {'use_tf32': '0'})]
    started = time.perf_counter()
    fixture = Fixture(args.model)
    session = ort.InferenceSession(str(args.model / 'model.onnx'), sess_options=options, providers=providers)
    load_ms = 1000 * (time.perf_counter() - started)
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA provider unavailable; refusing silent CPU fallback')
    print(json.dumps(dict(runtime=ort.__version__, providers=session.get_providers(),
                          provider_options=session.get_provider_options(), load_ms=load_ms)), flush=True)
    for fields in [1, 4]:
        started = time.perf_counter()
        probabilities = fixture.run(session, fields)
        first_ms = 1000 * (time.perf_counter() - started)
        for _ in range(args.warmup):
            fixture.run(session, fields)
        times = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            probabilities = fixture.run(session, fields)
            times.append(1000 * (time.perf_counter() - started))
        ordered = sorted(times)
        print(json.dumps(dict(language='python', runtime=ort.__version__, provider='CUDAExecutionProvider',
                              use_tf32=False, threads=args.threads, fields=fields,
                              shape=list(fixture.tensors(fields)['input_ids'].shape), warmup=args.warmup,
                              iterations=args.iterations, first_ms=first_ms,
                              p50_ms=ordered[math.ceil(.5 * len(times)) - 1],
                              p95_ms=ordered[math.ceil(.95 * len(times)) - 1],
                              samples_ms=times, probabilities=probabilities)), flush=True)
    del session
    # Profile a separate session so instrumentation does not distort timings.
    options.enable_profiling = True
    profile_directory = tempfile.mkdtemp(prefix='jevtpp-gpu-profile-')
    options.profile_file_prefix = str(Path(profile_directory) / 'ort')
    session = ort.InferenceSession(str(args.model / 'model.onnx'), sess_options=options, providers=providers)
    if session.get_providers()[0] != 'CUDAExecutionProvider':
        raise RuntimeError('CUDA provider unavailable in profiling session')
    fixture.run(session, 4)
    profile_path = session.end_profiling()
    events = json.loads(Path(profile_path).read_text())
    counts = Counter(event.get('args', {}).get('provider') for event in events if event.get('cat') == 'Node')
    model = onnx.load(str(args.model / 'model.onnx'))
    dtypes = Counter(onnx.TensorProto.DataType.Name(tensor.data_type) for tensor in model.graph.initializer)
    cpu_ops = Counter(event.get('args', {}).get('op_name') for event in events
                      if event.get('args', {}).get('provider') == 'CPUExecutionProvider')
    matrix_input_types = Counter(type_name for event in events
                                 if event.get('args', {}).get('op_name') in ('MatMul', 'FusedMatMul', 'Gemm')
                                 for tensor in event.get('args', {}).get('input_type_shape', [])
                                 for type_name in tensor)
    print(json.dumps(dict(profile=profile_path, profile_node_provider_counts=dict(counts),
                          cpu_op_event_counts=dict(cpu_ops), matrix_input_type_counts=dict(matrix_input_types),
                          weight_initializer_types=dict(dtypes))), flush=True)


if __name__ == '__main__':
    main()
