#!/usr/bin/env python3
"""Paired CPU benchmark for laya_latency.cpp; includes tokenization each call.

Run both languages serially on the same host/runtime, with MODEL THREADS
FIELDS ITERATIONS (FIELDS is 1 or 4; THREADS=0 selects the ORT default).
Session creation, first inference, then five warmups are excluded from p50/p95.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

CONTEXT = '''{
  "ticket": {
    "subject": "Production login is down",
    "message": "All our users are locked out. We have no workaround. Please fix this immediately!"
  },
  "customer": {"plan": "enterprise", "affected_users": 240},
  "service": {"status": "complete_outage", "sla_response_minutes": 30}
}'''
PREFIX = "Evaluate the incoming customer support ticket using the ticket, customer, and service context.\n\n"
QUESTIONS = [
    ("choice", "Which team should own this request?", ["Invoices, payments, or refund requests.", "Software bugs, crashes, or login failures.", "Upgrades, enterprise pricing, or new purchases.", "Unsolicited marketing or automated noise."]),
    ("noul", "Does this require attention within one hour?", []),
    ("score", "Rate the frustration level against the rubric.", ["Customer is patient and calm.", "Issue blocks work but has a workaround.", "Complete outage or severe frustration."]),
    ("noul", "The customer is angry.", []),
]


class Fixture:
    def __init__(self, root):
        self.agent = json.loads((root / "rl_agent_config.json").read_text())
        config = json.loads((root / "tokenizer/tokenizer_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer/tokenizer.json"))
        self.special = {key: self.tokenizer.token_to_id(config[key])
                        for key in ("cls_token", "sep_token", "mask_token", "pad_token")}
        self.mask_text = config["mask_token"]

    def tensors(self, fields):
        encode = lambda text: self.tokenizer.encode(text.replace(self.mask_text, " "), add_special_tokens=False).ids
        rows = []
        for kind, question, criteria in QUESTIONS[:fields]:
            options = (["false: no, the statement does not hold", "true: yes, the statement holds"] if kind == "noul"
                       else [f"level {i}: {text}" for i, text in enumerate(criteria)] if kind == "score" else criteria)
            head = encode(f"{kind} question: {PREFIX}{question}")
            option_ids = [[self.special["mask_token"], *encode(" " + option)[:48]] for option in options]
            budget = self.agent["head_max_len"] - sum(map(len, option_ids))
            if budget < 16:
                per = max(4, (self.agent["head_max_len"] - 16) // len(option_ids))
                option_ids = [value[:per] for value in option_ids]
                budget = self.agent["head_max_len"] - sum(map(len, option_ids))
            sequence = [self.special["cls_token"], *head[:max(8, budget)], self.special["sep_token"]]
            markers = []
            for value in option_ids:
                markers.append(len(sequence))
                sequence.extend(value)
            sequence.append(self.special["sep_token"])
            room = max(0, self.agent["max_len"] - len(sequence) - 1)
            sequence.extend(encode(CONTEXT)[:room])
            sequence.append(self.special["sep_token"])
            rows.append((sequence[:self.agent["max_len"]], markers, {"choice": 0, "score": 1, "noul": 2}[kind]))
        length = max(8, max(len(row[0]) for row in rows))
        width = max(2, max(len(row[1]) for row in rows))
        ids = np.full((fields, length), self.special["pad_token"], dtype=np.int64)
        attention = np.zeros_like(ids)
        positions = np.zeros((fields, width), dtype=np.int64)
        mask = np.zeros((fields, width), dtype=np.bool_)
        qtypes = np.zeros(fields, dtype=np.int64)
        for index, (sequence, markers, qtype) in enumerate(rows):
            ids[index, :len(sequence)] = sequence
            attention[index, :len(sequence)] = 1
            positions[index, :len(markers)] = markers
            mask[index, :len(markers)] = True
            qtypes[index] = qtype
        return dict(input_ids=ids, attention_mask=attention, marker_pos=positions, marker_mask=mask, qtype=qtypes)

    def run(self, session, fields):
        tensors = self.tensors(fields)
        logits = session.run(["logits"], tensors)[0]
        output = []
        for row, ((name, _, _), size, qtype) in enumerate(zip(QUESTIONS, [4, 2, 3, 2], tensors["qtype"])):
            bucket = "2" if size == 2 else "3-5"
            temperature = self.agent.get("temperature_by_options", {}).get(f"{name}:{bucket}", self.agent["temperature"][qtype])
            values = logits[row, :size]
            probabilities = np.exp((values - values.max()) / temperature)
            probabilities /= probabilities.sum()
            output.append(probabilities.tolist())
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("threads", type=int)
    parser.add_argument("fields", type=int, choices=[1, 4])
    parser.add_argument("iterations", type=int)
    parser.add_argument("--breakdown", action="store_true",
                        help="separately time tensor preparation, prepared inference and the full request")
    args = parser.parse_args()
    if args.threads < 0 or args.iterations < 1:
        parser.error("threads must be nonnegative and iterations positive")
    started = time.perf_counter()
    fixture = Fixture(args.model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(args.model / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - started) * 1000
    if args.breakdown:
        prepared = fixture.tensors(args.fields)
        output = {}
        for name, run, count in [
            ("tokenization_and_tensor_setup", lambda: fixture.tensors(args.fields), 200),
            ("session_run_prepared_tensors", lambda: session.run(["logits"], prepared), args.iterations),
            ("full_backend", lambda: fixture.run(session, args.fields), args.iterations),
        ]:
            for _ in range(5):
                run()
            samples = []
            for _ in range(count):
                started = time.perf_counter()
                run()
                samples.append((time.perf_counter() - started) * 1000)
            ordered = sorted(samples)
            output[name] = dict(p50_ms=ordered[math.ceil(.5 * count) - 1],
                                p95_ms=ordered[math.ceil(.95 * count) - 1], samples_ms=samples)
        print(json.dumps(output))
        return
    started = time.perf_counter()
    probabilities = fixture.run(session, args.fields)
    first_ms = (time.perf_counter() - started) * 1000
    for _ in range(5):
        fixture.run(session, args.fields)
    times = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        probabilities = fixture.run(session, args.fields)
        times.append((time.perf_counter() - started) * 1000)
    samples = times.copy()
    times.sort()
    print(json.dumps(dict(language="python", runtime=ort.__version__, threads=args.threads, fields=args.fields,
                         iterations=args.iterations, warmup=5, load_ms=load_ms, first_ms=first_ms,
                         p50_ms=times[math.ceil(.5 * len(times)) - 1], p95_ms=times[math.ceil(.95 * len(times)) - 1],
                         shape=list(fixture.tensors(args.fields)["input_ids"].shape), probabilities=probabilities,
                         samples_ms=samples)))


if __name__ == "__main__":
    main()
