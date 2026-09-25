#!/usr/bin/env python3
"""Compact Open-JEV requests and reproducible real-model latency measurements.

The serve command uses the pinned upstream loader. No model answers are faked;
JevT++ retains graph execution and controller ownership. Fewer questions change
the task, so latency ablations are not equivalent-quality gameplay benchmarks.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request
from urllib.parse import urlsplit

GOAL_CRITERIA = {
    "finish_fast": "Advance safely to the flag.",
    "stomp_enemy": "Defeat a nearby enemy from above.",
    "collect_powerup": "Collect a nearby mushroom or flower.",
    "collect_coins": "Collect nearby coins safely.",
    "score_attack": "Gain score from available targets.",
    "recover_momentum": "Escape a blocked position.",
}
TACTICAL_CRITERIA = {
    "right_run": "Run right on clear ground.",
    "right_run_jump": "Jump right over a hazard.",
    "right": "Move right at controlled speed.",
    "left": "Back away from a blocked position.",
}

def _number(value, default=999):
    return round(value, 1) if isinstance(value, (int, float)) and math.isfinite(value) else default

def compact_strategy_state(state: dict) -> dict:
    """Keep semantic facts; the full observation stays in the JevT++ graph."""
    player, terrain = state.get("player", {}), state.get("terrain", {})
    detectors, hazard = state.get("detectors", {}), state.get("hazard", {})
    return {
        "mode": state.get("strategy", {}).get("mode", "speedrun"),
        "grounded": bool(player.get("grounded")),
        "velocity": [_number(player.get("vx"), 0), _number(player.get("vy"), 0)],
        "enemy_distance": _number(hazard.get("enemy_distance")),
        "gap_distance": _number(terrain.get("gap_distance")),
        "obstacle_distance": _number(terrain.get("obstacle_distance")),
        "powerup_distance": _number(detectors.get("powerup_distance")),
        "coins_collected": _number(detectors.get("coins"), 0),
        "stalled": _number(state.get("recent_control", {}).get("stalled"), 0),
        "trajectory": state.get("trajectory", {}).get("committed_geometry", "none"),
        "support_reliability": terrain.get("observation_reliability", "unknown"),
    }

def compact_request(state: dict, *, tactical=False) -> dict:
    instruction = (
        "Choose the safe Mario controller action. Distances are pixels; 999 means absent."
        if tactical else
        "Choose Mario's goal for the selected mode.\n\nPreserve life. Distances are pixels; 999 means absent.")
    # JevT++ remote currently serializes enum members as option_N keys.
    criteria = TACTICAL_CRITERIA if tactical else GOAL_CRITERIA
    return {"model": "open-jev", "state": compact_strategy_state(state), "questions": {
        "q0": {"type": "choice", "instructions": instruction,
               "criteria": {f"option_{i}": value for i, value in enumerate(criteria.values())}}}}

def legacy_request(state: dict) -> dict:
    """Original C++ four-field request: 17 model sequences, 18 output bins."""
    prefix = "Control an original side-scrolling platform runner. Advance to the finish without falling or touching enemies.\n\n"
    goals = ["Maximize safe rightward progress and reach the flag quickly.",
             "Eliminate the nearest hostile by landing on it from above.",
             "Route toward a visible mushroom, flower, or star.",
             "Increase coins and score without sacrificing the current life.",
             "Prefer enemies, blocks, coins, and powerups that increase score.",
             "Escape a blocked position and restore rightward movement."]
    actions = ["Release controls and preserve the current trajectory.",
               "Walk right with controlled forward speed.",
               "Move right and begin or sustain a jump.",
               "Run right quickly on safe open ground.",
               "Run right while beginning or sustaining a long jump.",
               "Jump vertically without adding forward acceleration.",
               "Move left only to recover from a blocked or unsafe position."]
    def choice(instructions, criteria):
        return {"type": "choice", "instructions": prefix + instructions,
                "criteria": {f"option_{i}": value for i, value in enumerate(criteria)}}
    return {"model": "open-jev", "state": state, "questions": {
        "q0": choice("Choose the current tactical goal. Follow strategy.mode, detected targets, pace, score delta, and recent control outcome.", goals),
        "q1": choice("Choose the next controller macro. Respect projected contact time, jump phase, trusted terrain, and recent control outcome. A gap or obstacle within the reaction horizon requires a forward jump.", actions),
        "q2": {"type": "noul", "instructions": prefix + "Should a forward jump begin or remain held now? A near gap, obstacle, projected enemy contact, or an active gap crossing means yes."},
        "q3": {"type": "score", "instructions": prefix + "How dangerous is the immediate situation?", "criteria": [
            "Safe open movement with no immediate hazard.",
            "An obstacle, gap, or enemy will matter soon.",
            "Immediate collision, fall, or enemy threat."]}}}

def infer(url: str, request: dict) -> tuple[dict, float]:
    wire = urllib.request.Request(url.rstrip("/") + "/v1/systemone", json.dumps(request).encode(),
                                  {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(wire, timeout=60) as response:
        result = json.load(response)
    return result, (time.perf_counter() - started) * 1000

def _percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return round(values[lo] + (values[hi] - values[lo]) * (position - lo), 3)

def benchmark(args):
    rows = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
    states = [row["state"] for row in rows if row.get("state") and not row["state"].get("episode", {}).get("dead")]
    if not states:
        raise ValueError("Trace contains no live game states")
    chosen = [states[min(len(states) - 1, int(i * len(states) / args.samples))] for i in range(args.samples)]
    builders = {"legacy_four_questions": legacy_request,
                "compact_goal_six_candidates": compact_request,
                "compact_tactical_four_candidates": lambda state: compact_request(state, tactical=True)}
    output = {"protocol": {"trace": str(args.trace), "samples_per_variant": args.samples,
                           "warmups_per_variant": args.warmups,
                           "selection": "uniform indices over nonterminal states; variants round-robin",
                           "latency": "serial HTTP wall clock, warm service including tokenization and serialization",
                           "scope": "latency ablation, not gameplay or accuracy benchmark",
                           "cold_start": "excluded: load and CUDA/FLA compilation measured separately"},
              "endpoint": args.url, "variants": {}}
    samples = {name: [] for name in builders}
    for builder in builders.values():
        for _ in range(args.warmups):
            infer(args.url, builder(chosen[0]))
    for state in chosen:
        for name, builder in builders.items():
            request = builder(state)
            response, wall_ms = infer(args.url, request)
            samples[name].append({"wall_ms": round(wall_ms, 3),
                                  "request": request,
                                  "request_bytes": len(json.dumps(request).encode()),
                                  "state_bytes": len(json.dumps(request["state"]).encode()),
                                  "usage": response.get("usage"), "metadata": response.get("metadata"),
                                  "model": response.get("model"), "answers": response.get("answers")})
    for name, data in samples.items():
        latencies = [row["wall_ms"] for row in data]
        output["variants"][name] = {"wall_ms": {"p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95),
                                                       "mean": round(statistics.fmean(latencies), 3)}, "samples": data}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({name: report["wall_ms"] for name, report in output["variants"].items()}, indent=2))

def benchmark_strategy(args):
    """Measure the actual C++ route using unmodified graph-produced payloads."""
    states = []
    for line in args.trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("type") == "terminal":
            continue
        for node in row.get("graph", {}).get("nodes", []):
            state = node.get("output")
            if node.get("id") == "compact_state" and isinstance(state, dict) and "experience" in state:
                states.append(state)
    if not states:
        raise ValueError("Trace has no graph-produced compact_state nodes with experience")
    chosen = [states[min(len(states) - 1, int(i * len(states) / args.samples))] for i in range(args.samples)]

    def runtime():
        with urllib.request.urlopen(args.runtime_url.rstrip("/") + "/api/runtime", timeout=10) as response:
            data = json.load(response)
        evidence = data.get("runtime", {})
        if data.get("status") != "ready" or not evidence.get("validated"):
            raise RuntimeError("The model service has not verified its loaded CUDA tensors")
        if "4090" not in evidence.get("device_name", "") or evidence.get("parameter_devices") != [evidence.get("device")]:
            raise RuntimeError("Benchmark requires actual fully resident RTX 4090 CUDA model")
        return data

    def call(state):
        wire = urllib.request.Request(args.url.rstrip("/") + "/api/strategy", json.dumps(state).encode(),
                                      {"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(wire, timeout=60) as response:
            answer = json.load(response)
        elapsed = (time.perf_counter() - started) * 1000
        if not answer.get("ok"):
            raise RuntimeError(f"C++ strategy failed: {answer.get('error')}")
        return answer, elapsed

    initial_runtime = runtime()
    for _ in range(args.warmups):
        call(chosen[0])
    samples = []
    for state in chosen:
        answer, elapsed = call(state)
        evidence = runtime()
        actual = evidence.get("last_inference") or {}
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        if actual.get("state_sha256") != hashlib.sha256(encoded.encode()).hexdigest():
            raise RuntimeError("Runtime counters do not match this state; another caller may be using the model")
        if actual.get("questions") != 1 or actual.get("metadata", {}).get("candidate_sequences") != 6:
            raise RuntimeError("Production route did not issue the expected single six-candidate Choice")
        cache = actual["metadata"].get("prefix_cache", {})
        # One shared prefill and one padded suffix batch determine the largest
        # complete tokenized candidate without loading a separate tokenizer.
        max_tokens = None
        if cache.get("prefill_calls") == 1 and cache.get("suffix_batches") == 1 and cache.get("max_suffix_batch"):
            max_tokens = cache["shared_prefix_tokens"] + cache["suffix_tokens"] // cache["max_suffix_batch"]
        samples.append({"wall_ms": round(elapsed, 3), "state": state, "answer": answer,
                        "actual_inference": actual, "maximum_candidate_tokens": max_tokens,
                        "gpu_memory": evidence["runtime"]["memory"]})
    latencies = [sample["wall_ms"] for sample in samples]
    report = {
        "protocol": {"route": args.url.rstrip("/") + "/api/strategy", "trace": str(args.trace),
                     "runtime_route": args.runtime_url.rstrip("/") + "/api/runtime",
                     "measured_calls": len(samples), "excluded_warmups": args.warmups,
                     "payload": "unaltered C++ graph compact_state including bounded experience context",
                     "scope": "warm serial C++ HTTP to Open-JEV CUDA to typed JevT++ result; not win rate",
                     "latency": "end-to-end HTTP wall clock; runtime inspection excluded", "concurrency": 1},
        "initial_runtime": initial_runtime,
        "wall_ms": {"p50": _percentile(latencies, .5), "p95": _percentile(latencies, .95),
                    "mean": round(statistics.fmean(latencies), 3)},
        "input_token_range": [min(s["actual_inference"]["usage"]["input_tokens"] for s in samples),
                              max(s["actual_inference"]["usage"]["input_tokens"] for s in samples)],
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wall_ms": report["wall_ms"], "input_token_range": report["input_token_range"]}, indent=2))


def require_cuda_device(torch, requested_device, expected_device_name="RTX 4090"):
    """Refuse an unavailable, CPU, or unexpected GPU before loading weights."""
    device = torch.device(requested_device)
    if device.type != "cuda":
        raise RuntimeError(f"Open-JEV requires CUDA inference; refusing device {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("Open-JEV requires CUDA; CPU fallback is disabled")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device index {index} is unavailable")
    name = torch.cuda.get_device_name(index)
    expected = " ".join(str(expected_device_name).lower().split())
    if not expected or expected not in " ".join(name.lower().split()):
        raise RuntimeError(f"Expected GPU {expected_device_name!r}; CUDA device {index} is {name!r}")
    return f"cuda:{index}", name


def inspect_cuda_runtime(torch, predictor, requested_device, expected_device_name="RTX 4090"):
    """Inspect real loaded tensors, including the trained scalar decision head."""
    device, name = require_cuda_device(torch, requested_device, expected_device_name)
    model = predictor.scorer.model
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    if not parameters:
        raise RuntimeError("Loaded Open-JEV model has no inspectable parameters")
    invalid = [(key, str(value.device)) for key, value in parameters + buffers
               if str(value.device) != device]
    if invalid:
        details = ", ".join(f"{key}={where}" for key, where in invalid[:6])
        raise RuntimeError(f"Open-JEV tensors are not fully on {device}: {details}; CPU/offload fallback is disabled")
    if str(model.device_name) != device:
        raise RuntimeError(f"Open-JEV input device {model.device_name} does not match tensor device {device}")
    return {
        "validated": True,
        "cuda_available": True,
        "device": device,
        "device_name": name,
        "expected_device_name": expected_device_name,
        "parameter_devices": sorted({str(value.device) for _, value in parameters}),
        "parameter_dtypes": sorted({str(value.dtype) for _, value in parameters}),
        "parameter_count": sum(value.numel() for _, value in parameters),
        "buffer_devices": sorted({str(value.device) for _, value in buffers}),
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "memory": {
            "process_allocated_bytes": torch.cuda.memory_allocated(device),
            "process_reserved_bytes": torch.cuda.memory_reserved(device),
            "process_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        },
        "checkpoint": dict(predictor.provenance),
    }


def monitor_inference(predictor):
    """Retain real scoring counters; never substitute or reuse model answers."""
    original_predict = predictor.predict
    recent = {"last_inference": None}
    sequence = 0

    def predict(request):
        nonlocal sequence
        result = original_predict(request)
        sequence += 1
        state = json.dumps(request["state"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        recent["last_inference"] = {
            "sequence": sequence,
            "state_sha256": hashlib.sha256(state.encode()).hexdigest(),
            "state_bytes": len(state.encode()),
            "questions": len(request["questions"]),
            "model": result.get("model"),
            "usage": result.get("usage"),
            "metadata": result.get("metadata"),
        }
        return result

    predictor.predict = predict
    return recent


def install_runtime_endpoint(server, predictor, inspect, recent=None):
    """Attach device evidence to the exact server serving JevT++ inference."""
    upstream_handler = server.RequestHandlerClass

    class RuntimeHandler(upstream_handler):
        def do_GET(self):
            if urlsplit(self.path).path not in ("/api/runtime", "/health"):
                return super().do_GET()
            try:
                runtime = inspect()
            except RuntimeError as error:
                return self.send(503, {"status": "error", "error": str(error),
                                       "runtime": {"validated": False, "cpu_fallback": False}})
            payload = {"status": "ready", "model": predictor.model_name,
                       "method": predictor.method, "runtime": runtime}
            if recent is not None:
                payload["last_inference"] = recent["last_inference"]
            return self.send(200, payload)

    server.RequestHandlerClass = RuntimeHandler


def serve(args):
    os.environ.setdefault("JEV_RAGGED_SUFFIX", "1")
    # This opt-in is unmeasured here; compact single-question requests do not
    # have a separate question prefill in the first place.
    os.environ.setdefault("JEV_NO_QPREFILL", "1" if args.no_question_prefill else "0")
    os.environ.setdefault("JEV_PROFILE", "1" if args.profile else "0")
    local_cache = Path(__file__).resolve().parents[2] / "models" / "huggingface-cache"
    if local_cache.is_dir():
        os.environ.setdefault("HF_HOME", str(local_cache))
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    from jev.server import make_server
    from jev.serving import load_predictor
    device, _ = require_cuda_device(torch, args.device, args.expected_device_name)
    started = time.perf_counter()
    predictor = load_predictor(checkpoint=str(args.checkpoint), device=device,
                               max_length=args.max_length, batch_size=args.batch_size,
                               prefix_cache=not args.uncached)
    def inspect():
        return inspect_cuda_runtime(torch, predictor, device, args.expected_device_name)
    runtime = inspect()  # Validate actual weights before opening the HTTP port.
    recent = monitor_inference(predictor)
    server = make_server(predictor, "127.0.0.1", args.port)
    install_runtime_endpoint(server, predictor, inspect, recent)
    print(json.dumps({"url": f"http://127.0.0.1:{args.port}", "model": predictor.model_name,
                      "method": predictor.method, "gpu": runtime["device_name"],
                      "runtime": runtime,
                      "load_seconds": round(time.perf_counter() - started, 3),
                      "prefix_cache": not args.uncached,
                      "ragged_suffix": os.environ["JEV_RAGGED_SUFFIX"],
                      "no_question_prefill": os.environ["JEV_NO_QPREFILL"],
                      "provenance": predictor.provenance}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b = commands.add_parser("benchmark")
    b.add_argument("--url", default="http://127.0.0.1:8791")
    b.add_argument("--trace", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--samples", type=int, default=10)
    b.add_argument("--warmups", type=int, default=2)
    actual = commands.add_parser("benchmark-strategy")
    actual.add_argument("--url", default="http://127.0.0.1:4174")
    actual.add_argument("--runtime-url", default="http://127.0.0.1:8791")
    actual.add_argument("--trace", type=Path, required=True)
    actual.add_argument("--output", type=Path, required=True)
    actual.add_argument("--samples", type=int, default=16)
    actual.add_argument("--warmups", type=int, default=3)
    s = commands.add_parser("serve")
    s.add_argument("--source", type=Path, default=Path(".deps/open-jev-src"))
    s.add_argument("--checkpoint", type=Path, default=Path("models/open-jev-2b/package/checkpoint"))
    s.add_argument("--device", default="cuda:0")
    s.add_argument("--expected-device-name", default="RTX 4090")
    s.add_argument("--port", type=int, default=8791)
    s.add_argument("--max-length", type=int, default=4096)
    s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--profile", action="store_true")
    s.add_argument("--no-question-prefill", action="store_true")
    s.add_argument("--uncached", action="store_true")
    args = parser.parse_args()
    if args.command in ("benchmark", "benchmark-strategy"):
        if args.samples < 1 or args.warmups < 0:
            parser.error("samples must be positive; warmups cannot be negative")
        (benchmark_strategy if args.command == "benchmark-strategy" else benchmark)(args)
    else:
        serve(args)

if __name__ == "__main__":
    main()
