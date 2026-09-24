#!/usr/bin/env python3
"""Compare opt-in precision candidates to FP32; built-in labels are smoke tests only."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort

from laya_latency import CONTEXT, PREFIX, QUESTIONS, Fixture


# Exact states and labels from laya_quality.cpp. These eight hand-written labels
# are a regression smoke fixture, not representative business-quality evidence.
SMOKE_CASES = [
    ("My card was charged twice. Please refund the duplicate payment.", 0, False),
    ("Production login is down for every employee and there is no workaround.", 1, True),
    ("Please quote the enterprise plan for 500 seats next quarter.", 2, False),
    ("BUY FOLLOWERS NOW!!! Visit our promotion link for guaranteed growth.", 3, False),
    ("The mobile app crashes whenever I upload an invoice.", 1, False),
    ("Our annual invoice has the wrong tax number and is due tomorrow.", 0, True),
    ("We want to upgrade today; procurement needs pricing within an hour.", 2, True),
    ("Automated crypto investment offer. Limited time, click this link.", 3, False),
]


def build_tensors(fixture, state, questions, prefix):
    encode = lambda text: fixture.tokenizer.encode(text.replace(fixture.mask_text, " "), add_special_tokens=False).ids
    rows, sizes = [], []
    for kind, question, criteria in questions:
        options = (["false: no, the statement does not hold", "true: yes, the statement holds"] if kind == "noul"
                   else [f"level {i}: {text}" for i, text in enumerate(criteria)] if kind == "score" else criteria)
        head = encode(f"{kind} question: {prefix}{question}")
        option_ids = [[fixture.special["mask_token"], *encode(" " + text)[:48]] for text in options]
        budget = fixture.agent["head_max_len"] - sum(map(len, option_ids))
        if budget < 16:
            per = max(4, (fixture.agent["head_max_len"] - 16) // len(option_ids))
            option_ids = [value[:per] for value in option_ids]
            budget = fixture.agent["head_max_len"] - sum(map(len, option_ids))
        sequence = [fixture.special["cls_token"], *head[:max(8, budget)], fixture.special["sep_token"]]
        markers = []
        for value in option_ids:
            markers.append(len(sequence))
            sequence.extend(value)
        sequence.append(fixture.special["sep_token"])
        room = max(0, fixture.agent["max_len"] - len(sequence) - 1)
        sequence.extend(encode(state)[:room])
        sequence.append(fixture.special["sep_token"])
        rows.append((sequence[:fixture.agent["max_len"]], markers, {"choice": 0, "score": 1, "noul": 2}[kind]))
        sizes.append(len(options))
    length = max(8, max(len(row[0]) for row in rows))
    width = max(2, max(sizes))
    ids = np.full((len(rows), length), fixture.special["pad_token"], dtype=np.int64)
    attention = np.zeros_like(ids)
    positions = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.zeros_like(positions, dtype=np.bool_)
    qtypes = np.zeros(len(rows), dtype=np.int64)
    for index, (sequence, markers, qtype) in enumerate(rows):
        if any(position >= len(sequence) for position in markers):
            raise ValueError("question criteria exceed context budget")
        ids[index, :len(sequence)] = sequence
        attention[index, :len(sequence)] = 1
        positions[index, :len(markers)] = markers
        mask[index, :len(markers)] = True
        qtypes[index] = qtype
    return dict(input_ids=ids, attention_mask=attention, marker_pos=positions,
                marker_mask=mask, qtype=qtypes), sizes


def probabilities(logits, sizes, questions, agent):
    answers = []
    for row, (size, (kind, _, _)) in enumerate(zip(sizes, questions)):
        bucket = "2" if size <= 2 else "3-5" if size <= 5 else "6-10" if size <= 10 else "11+"
        temperature = agent.get("temperature_by_options", {}).get(
            f"{kind}:{bucket}", agent["temperature"][{"choice": 0, "score": 1, "noul": 2}[kind]])
        values = np.asarray(logits[row, :size], dtype=np.float64)
        if not np.all(np.isfinite(values)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("non-finite logits or invalid calibration temperature")
        values = np.exp((values - values.max()) / temperature)
        answers.append((values / values.sum()).tolist())
    return answers


def comparison(reference, candidate):
    if len(reference) != len(candidate) or any(len(a) != len(b) for a, b in zip(reference, candidate)):
        raise ValueError("candidate answer shape differs from reference")
    deltas = [abs(float(a) - float(b)) for left, right in zip(reference, candidate)
              for a, b in zip(left, right)]
    return dict(max_probability_delta=max(deltas, default=0.0),
                mean_probability_delta=sum(deltas) / max(1, len(deltas)),
                argmax_disagreements=sum(int(np.argmax(a) != np.argmax(b))
                                        for a, b in zip(reference, candidate)),
                compared_fields=len(reference))


def quality(outputs, labels):
    if not labels:
        return None
    category_correct, urgent_correct, category_nll, urgent_brier = 0, 0, 0.0, 0.0
    for output, (_, category, urgent) in zip(outputs, labels):
        category_correct += int(np.argmax(output[0]) == category)
        urgent_correct += int((output[1][1] >= 0.5) == urgent)
        category_nll -= math.log(max(1e-12, output[0][category]))
        urgent_brier += (output[1][1] - float(urgent)) ** 2
    count = len(labels)
    return dict(examples=count, category_accuracy=category_correct / count,
                urgent_accuracy=urgent_correct / count, category_nll=category_nll / count,
                urgent_brier=urgent_brier / count)


def gate_failures(metrics, quality_metrics, reference_quality, gates):
    failures = []
    for argument, metric in [("max_probability_delta", "max_probability_delta"),
                             ("max_argmax_disagreements", "argmax_disagreements")]:
        limit = gates.get(argument)
        if limit is not None and metrics[metric] > limit:
            failures.append(f"{metric}={metrics[metric]} exceeds {limit}")
    for field in ("category", "urgent"):
        minimum = gates.get(f"min_{field}_accuracy")
        if minimum is not None and quality_metrics[f"{field}_accuracy"] < minimum:
            failures.append(f"{field}_accuracy={quality_metrics[f'{field}_accuracy']} below {minimum}")
        limit = gates.get("max_accuracy_drop")
        if limit is not None:
            drop = reference_quality[f"{field}_accuracy"] - quality_metrics[f"{field}_accuracy"]
            if drop > limit:
                failures.append(f"{field}_accuracy_drop={drop} exceeds {limit}")
    return failures


def percentile(values, q):
    return sorted(values)[math.ceil(q * len(values)) - 1]


def model_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="original FP32-compute bundle/config/tokenizer")
    parser.add_argument("--candidate", type=Path, action="append", required=True, help="candidate ONNX path; repeatable")
    parser.add_argument("--reference", type=Path, help="defaults to BUNDLE/model.onnx")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--labels", type=Path, help="JSON list of {state, category:0..3, urgent:bool}; replaces smoke labels")
    parser.add_argument("--output", type=Path, help="write report to a new JSON file")
    parser.add_argument("--max-probability-delta", type=float)
    parser.add_argument("--max-argmax-disagreements", type=int)
    parser.add_argument("--min-category-accuracy", type=float)
    parser.add_argument("--min-urgent-accuracy", type=float)
    parser.add_argument("--max-accuracy-drop", type=float)
    args = parser.parse_args()
    if args.threads < 1 or args.iterations < 1 or args.warmup < 0:
        parser.error("threads/iterations must be positive and warmup nonnegative")
    gate_names = ["max_probability_delta", "max_argmax_disagreements", "min_category_accuracy", "min_urgent_accuracy", "max_accuracy_drop"]
    gates = {name: getattr(args, name) for name in gate_names}
    for name, value in gates.items():
        if value is not None and (not math.isfinite(value) or value < 0 or (name != "max_argmax_disagreements" and value > 1)):
            parser.error(f"invalid gate: {name}")
    if args.output and args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = dict(runtime=ort.__version__, provider=args.provider, use_tf32=False,
                  timing_scope="ORT session.run with prepared tensors; synchronous CPU output copies included",
                  threads=args.threads, iterations=args.iterations, warmup=args.warmup,
                  quality_dataset="eight hand-written smoke cases; not a business benchmark",
                  gates=gates, candidates=[], default_backend_changed=False)
    failed = False
    try:
        labels = SMOKE_CASES
        if args.labels:
            records = json.loads(args.labels.read_text())
            if not isinstance(records, list) or not records:
                raise ValueError("labels must be a nonempty JSON list")
            labels = []
            for record in records:
                if (not isinstance(record.get("state"), str) or type(record.get("category")) is not int
                        or not 0 <= record["category"] <= 3 or type(record.get("urgent")) is not bool):
                    raise ValueError("each label needs string state, category integer 0..3 and boolean urgent")
                labels.append((record["state"], record["category"], record["urgent"]))
            report["quality_dataset"] = dict(path=str(args.labels), sha256=model_hash(args.labels), examples=len(labels))
        fixture = Fixture(args.bundle)
        cases = [(state, QUESTIONS[:2], "Evaluate the incoming customer support ticket.\n\n") for state, _, _ in labels]
        cases += [(CONTEXT, QUESTIONS, PREFIX),
                  ('{"ticket":"Nie mogę się zalogować. Cała firma jest zablokowana.","workaround":false}', QUESTIONS, PREFIX)]
        prepared = [(build_tensors(fixture, state, questions, prefix), questions) for state, questions, prefix in cases]
        report["tensor_shapes"] = [list(item[0][0]["input_ids"].shape) for item in prepared]
        if args.provider == "cuda":
            ort.preload_dlls()
        expected_provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
        if expected_provider not in ort.get_available_providers():
            raise RuntimeError(f"required provider unavailable: {expected_provider}")
        providers = [(expected_provider, {"use_tf32": "0"})] if args.provider == "cuda" else [expected_provider]

        def evaluate(path):
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            started = time.perf_counter()
            session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
            if session.get_providers()[0] != expected_provider:
                raise RuntimeError("requested provider failed to initialize; refusing silent fallback")
            if next(item.type for item in session.get_outputs() if item.name == "logits") != "tensor(float)":
                raise RuntimeError("public logits must remain float32")
            load_ms = (time.perf_counter() - started) * 1000
            result = dict(path=str(path), sha256=model_hash(path), providers=session.get_providers(),
                          provider_options=session.get_provider_options(), load_ms=load_ms)
            outputs, samples = [], []
            first_started = time.perf_counter()
            for (tensors, sizes), questions in prepared:
                logits = session.run(["logits"], tensors)[0]
                outputs.append(probabilities(logits, sizes, questions, fixture.agent))
            result["first_suite_ms"] = (time.perf_counter() - first_started) * 1000
            for (tensors, _), _ in prepared:
                for _ in range(args.warmup):
                    session.run(["logits"], tensors)
            for _ in range(args.iterations):
                for (tensors, _), _ in prepared:
                    started = time.perf_counter()
                    session.run(["logits"], tensors)
                    samples.append((time.perf_counter() - started) * 1000)
            result.update(p50_ms=percentile(samples, .5), p95_ms=percentile(samples, .95),
                          p99_ms=percentile(samples, .99), samples_ms=samples,
                          quality=quality(outputs[:len(labels)], labels), probabilities=outputs)
            del session
            gc.collect()
            return result

        reference = evaluate(args.reference or args.bundle / "model.onnx")
        report["reference"] = reference
        reference_flat = [row for case in reference["probabilities"] for row in case]
        for candidate in args.candidate:
            try:
                measured = evaluate(candidate)
                measured.update(comparison(reference_flat, [row for case in measured["probabilities"] for row in case]))
                measured["gate_failures"] = gate_failures(measured, measured["quality"], reference["quality"], gates)
                measured["status"] = "failed" if measured["gate_failures"] else "passed" if any(value is not None for value in gates.values()) else "measured_no_gates"
                failed |= bool(measured["gate_failures"])
                report["candidates"].append(measured)
            except Exception as error:
                failed = True
                report["candidates"].append(dict(path=str(candidate), status="failed", error=f"{type(error).__name__}: {error}"))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        failed = True
    report["status"] = "failed" if failed else "complete"
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(rendered + "\n")
    print(rendered, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
