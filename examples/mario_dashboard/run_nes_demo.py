#!/usr/bin/env python3
"""Run the real NES emulator loop with JevT++ as the decision service."""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from nes_state import StateTracker, unwrap_ram


ACTIONS = ("noop", "right", "right_jump", "right_run", "right_run_jump", "jump", "left", "left_jump")
ACTION_INDEX = {name: index for index, name in enumerate(ACTIONS)}
JUMP_RELEASE_INDEX = {"right_jump": 1, "right_run_jump": 3, "jump": 0, "left_jump": 6}


@dataclass
class LiveState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame: bytes = b""
    payload: dict[str, Any] = field(default_factory=lambda: {"status": "booting"})
    command: str | None = None

    def update(self, *, frame: bytes | None = None, payload: dict[str, Any] | None = None) -> None:
        with self.lock:
            if frame is not None:
                self.frame = frame
            if payload is not None:
                self.payload = payload


def encode_frame(frame: Any) -> bytes:
    from PIL import Image
    image = Image.fromarray(frame).convert("RGB")
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


class DashboardHandler(SimpleHTTPRequestHandler):
    live: LiveState
    web_root: Path

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].lstrip("/") or "index.html"
        candidate = (self.web_root / clean).resolve()
        return str(candidate if candidate.is_relative_to(self.web_root.resolve()) else self.web_root / "__not_found__")

    def end_headers(self) -> None:
        # This is a live development/telemetry surface. A stale index keeps an
        # old graph DOM even though /api/live is current, which is especially
        # confusing while iterating on detector nodes.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/api/live"):
            with self.live.lock:
                body = json.dumps(self.live.payload, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/api/frame"):
            with self.live.lock:
                body = self.live.frame
            self.send_response(200 if body else 204)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/command":
            self.send_error(404); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("invalid command size")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("command must be an object")
        except (ValueError, json.JSONDecodeError):
            self.send_error(400); return
        command = body.get("command")
        if command == "model":
            with self.live.lock:
                options=self.live.payload.get("model_selection",{}).get("options",[])
                available=any(item.get("id")==body.get("model") and item.get("available") for item in options)
                if available: self.live.command=f"model:{body['model']}"
            self.send_response(204 if available else 409);self.end_headers();return
        if command in {"pause", "resume", "restart"}:
            with self.live.lock:
                self.live.command = command
            self.send_response(204); self.end_headers(); return
        if command == "mode" and body.get("mode") in {"speedrun", "hunter", "collector", "score_attack"}:
            with self.live.lock:
                self.live.command = f"mode:{body['mode']}"
            self.send_response(204); self.end_headers(); return
        if command == "knowledge_scope" and body.get("scope") in {"run", "session"}:
            with self.live.lock:
                self.live.command = f"knowledge_scope:{body['scope']}"
            self.send_response(204); self.end_headers(); return
        self.send_error(400)

    def log_message(self, *_: Any) -> None:
        return


def call_model(url: str, state: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(state, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=65) as response:
        return json.load(response)


def start_engine(args: argparse.Namespace, base: Path) -> subprocess.Popen[str] | None:
    try:
        with urllib.request.urlopen(args.model_url.replace("/api/decision", "/api/health"), timeout=0.5) as response:
            health = json.load(response)
        expected = "Open-JEV" if args.open_jev_url else "Laya" if args.model else "preview"
        if health.get("protocol") != 2 or expected not in health.get("backend", ""):
            raise RuntimeError("Existing engine has incompatible protocol/backend; restart it explicitly")
        if getattr(args, "model_policy", "plan") == "plan" and "plan_rank_v1" not in health.get("capabilities", []):
            raise RuntimeError("Existing engine lacks plan ranking; restart it explicitly")
        return None
    except (urllib.error.URLError, json.JSONDecodeError):
        pass
    command = [str(Path(args.engine).resolve()), "--port", str(args.engine_port),
               "--web-root", str((base / "web").resolve())]
    if args.model:
        command += ["--model", str(Path(args.model).resolve())]
    if args.open_jev_url:
        command += ["--open-jev-url", args.open_jev_url]
    environment = os.environ.copy()
    repository = base.parents[1]
    runtime_paths = glob.glob(str(repository / ".deps" / "onnxruntime-linux-x64-gpu*" / "lib"))
    runtime_paths += glob.glob(str(repository / ".venv-mario" / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib"))
    if runtime_paths:
        environment["LD_LIBRARY_PATH"] = ":".join(runtime_paths + [environment.get("LD_LIBRARY_PATH", "")])
    process = subprocess.Popen(command, text=True, env=environment)
    # First CUDA session creation can take longer on WSL while kernels and the
    # execution plan are initialized. Do not mistake a healthy cold start for
    # a failed backend.
    for _ in range(300):
        if process.poll() is not None:
            raise RuntimeError("JevT++ decision service exited during startup")
        try:
            urllib.request.urlopen(args.model_url.replace("/api/decision", "/"), timeout=0.2)
            return process
        except Exception:
            time.sleep(0.1)
    process.terminate()
    raise RuntimeError("JevT++ decision service did not become ready")


def create_environment(env_id: str) -> Any:
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace
    except ImportError as error:
        raise RuntimeError("Install the NES extras from requirements-mario.txt") from error
    try:
        environment = gym_super_mario_bros.make(env_id, render_mode="rgb_array", apply_api_compatibility=True)
    except TypeError:
        environment = gym_super_mario_bros.make(env_id)
    return JoypadSpace(environment, [*SIMPLE_MOVEMENT, ["left", "A"]])


def native_info(value: Any) -> Any:
    """Normalize NumPy scalars at the emulator boundary for logs and HTTP."""
    if isinstance(value, dict):
        return {key: native_info(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_info(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return native_info(value.item())
    return value


def reset_environment(environment: Any, seed: int) -> tuple[Any, dict[str, Any]]:
    try:
        result = environment.reset(seed=seed)
    except TypeError:
        if hasattr(environment, "seed"):
            environment.seed(seed)
        result = environment.reset()
    if isinstance(result, tuple):
        return result[0], native_info(result[1])
    frame, _, done, info = environment.step(0)
    if done:
        result = environment.reset()
        frame = result[0] if isinstance(result, tuple) else result
    return frame, native_info(info)


def step_environment(environment: Any, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    result = environment.step(action)
    if len(result) == 5:
        frame, reward, terminated, truncated, info = result
        return frame, float(reward), bool(terminated), bool(truncated), native_info(info)
    frame, reward, done, info = result
    return frame, float(reward), bool(done), False, native_info(info)


def current_environment_info(environment: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    """Read state after gym's internal death/stage transition skip.

    step_environment returns the event info captured before `_did_step`; the
    rendered frame and RAM can already belong to the next stage or life.
    Keep the two snapshots separate so a transient flag/death is not lost.
    """
    # Gym's EnvCompatibility.unwrapped is still EnvCompatibility, not the
    # NESpy environment. Its missing _get_info previously returned the
    # pre-transition event_info after a death or stage clear. Walk the actual
    # wrapper chain, as the RAM reader does.
    current_environment = environment
    visited: set[int] = set()
    while id(current_environment) not in visited:
        visited.add(id(current_environment))
        reader = getattr(current_environment, "_get_info", None)
        if callable(reader):
            current = native_info(reader())
            return current if isinstance(current, dict) else dict(fallback)
        current_environment = getattr(current_environment, "env", None)
        if current_environment is None:
            break
    return dict(fallback)


def run(args: argparse.Namespace) -> None:
    base = Path(__file__).resolve().parent
    engine = start_engine(args, base)
    live = LiveState()
    DashboardHandler.live = live
    DashboardHandler.web_root = base / "web"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.dashboard_port), DashboardHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="mario-dashboard").start()
    print(f"Dashboard: http://127.0.0.1:{args.dashboard_port}/")

    environment = create_environment(args.env)
    tracker = StateTracker(args.frames_per_decision)
    frame, info = reset_environment(environment, args.seed)
    active_action = "noop"
    decision: dict[str, Any] = {"action": active_action, "probabilities": {name: 0 for name in ACTIONS}}
    previous_reward = 0.0
    response_delay = 0
    reward_total = 0.0
    paused = False
    decision_index = 0
    frame_index = 0
    objective_mode = args.mode
    artifacts = base / "artifacts"
    artifacts.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = artifacts / f"run-{stamp}.jsonl"

    try:
        with log_path.open("w", encoding="utf-8") as log:
            while args.max_decisions == 0 or decision_index < args.max_decisions:
                with live.lock:
                    command, live.command = live.command, None
                if command == "pause": paused = True
                elif command == "resume": paused = False
                elif command == "restart":
                    frame, info = reset_environment(environment, args.seed)
                    tracker.reset(); active_action = "noop"; decision_index = frame_index = 0
                    reward_total = previous_reward = 0.0
                    paused = False
                elif command and command.startswith("mode:"):
                    objective_mode = command.split(":", 1)[1]
                    if paused:
                        frame, info = reset_environment(environment, args.seed)
                        tracker.reset(); active_action = "noop"; decision_index = frame_index = 0
                        reward_total = previous_reward = 0.0
                        paused = False
                if paused:
                    time.sleep(0.03); continue

                model_state, debug = tracker.parse(
                    info, unwrap_ram(environment), previous_action=active_action,
                    previous_reward=previous_reward, response_delay_frames=response_delay,
                    objective_mode=objective_mode,
                )
                decision_updated = False
                previous_active_action = active_action
                if frame_index % args.frames_per_decision == 0:
                    requested_at = time.perf_counter()
                    # Checkers are deterministic local evidence for the JevT++
                    # compositor and dashboard. Repeating their verbose labels
                    # inside the learned model context roughly doubles tokens
                    # without adding information already present in terrain,
                    # hazard and trajectory fields.
                    inference_state = dict(model_state)
                    inference_state.pop("checker_bus", None)
                    decision = call_model(args.model_url, inference_state)
                    if not decision.get("ok"):
                        raise RuntimeError(decision.get("error", "model decision failed"))
                    active_action = str(decision["action"])
                    decision_updated = True
                    # This legacy loop freezes the emulator during inference.
                    # Wall latency cannot be converted to simulated state age.
                    response_delay = 0
                    record = {**decision, "decision": decision_index, "state": model_state, "debug": debug,
                              "episode_reward": reward_total}
                    log.write(json.dumps(record, separators=(",", ":")) + "\n"); log.flush()
                    decision_index += 1

                payload = {
                    "status": "paused" if paused else "running", "decision_index": decision_index,
                    "frame_index": frame_index, "action": active_action, "decision": decision,
                    "state": model_state, "debug": debug, "episode_reward": reward_total,
                    "env": args.env, "frames_per_decision": args.frames_per_decision,
                    "mode": objective_mode,
                    "decision_age_frames": frame_index % args.frames_per_decision,
                    "frames_until_decision": (args.frames_per_decision - frame_index % args.frames_per_decision) % args.frames_per_decision,
                }
                live.update(frame=encode_frame(frame), payload=payload)
                step_action = ACTION_INDEX[active_action]
                if (decision_updated and active_action == previous_active_action and
                        active_action in JUMP_RELEASE_INDEX and model_state["player"]["grounded"]):
                    step_action = JUMP_RELEASE_INDEX[active_action]
                frame, reward, terminated, truncated, info = step_environment(environment, step_action)
                previous_reward = reward; reward_total += reward; frame_index += 1
                if terminated or truncated or model_state["episode"]["dead"] or model_state["episode"]["stage_clear"]:
                    payload["status"] = "clear" if model_state["episode"]["stage_clear"] else "ended"
                    live.update(frame=encode_frame(frame), payload=payload)
                    paused = True
                time.sleep(max(0.0, 1 / args.fps))
    finally:
        environment.close(); httpd.shutdown()
        if engine is not None:
            engine.terminate()
        print(f"Run log: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="SuperMarioBros-v0", help="full NES game by default; individual stages remain available for tests")
    parser.add_argument("--engine", default="build-mario/examples/jevt_mario_demo")
    parser.add_argument("--engine-port", type=int, default=4174)
    parser.add_argument("--dashboard-port", type=int, default=4173)
    parser.add_argument("--model-url", help="defaults to the selected engine port")
    parser.add_argument("--model", help="Laya ONNX model directory; enables the required CUDA backend")
    parser.add_argument("--laya-model", help="Additional LAYA CUDA bundle for the live model selector (engine port + 1)")
    parser.add_argument("--open-jev-url", help="local Open-JEV service root used only as the JevT++ model backend")
    parser.add_argument("--frames-per-decision", type=int, default=30, help="minimum model-intent interval; fast controller runs every frame")
    parser.add_argument("--control", choices=("async", "legacy"), default="async")
    parser.add_argument("--controller", help="persistent JevT++ graph executable")
    parser.add_argument("--model-policy", choices=("plan","goal"), default="plan", help="Rank actual bounded plans; goal retains the v13 comparison path")
    parser.add_argument("--dashboard-fps", type=float, default=12)
    parser.add_argument("--max-frames", type=int, default=0, help="optional diagnostic frame limit; 0 means play until game over or full completion")
    parser.add_argument("--exit-on-terminal", action="store_true")
    parser.add_argument("--max-decisions", type=int, default=0, help="optional model-decision limit; 0 means unlimited")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fresh-memory", action="store_true", help="Diagnostic stage runs: do not import or persist experience")
    parser.add_argument("--knowledge-scope", choices=("run", "session"), default="run",
                        help="Run resets memory at GAME OVER; session retains observed map and ontology across new games in this process")
    parser.add_argument("--trace-every-frame", action="store_true", help="Diagnostic full-action trace; larger disk usage than bounded failure windows")
    parser.add_argument("--mode", choices=("speedrun", "hunter", "collector", "score_attack"), default="speedrun")
    args = parser.parse_args()
    if args.model_url is None:
        args.model_url = f"http://127.0.0.1:{args.engine_port}/api/decision"
    if min(args.fps, args.dashboard_fps, args.frames_per_decision) <= 0 or args.max_frames < 0 or args.max_decisions < 0:
        parser.error("frame rates must be positive; max-frames and max-decisions may be zero")
    if args.control == "legacy" and args.env in {f"SuperMarioBros-v{version}" for version in range(4)}:
        parser.error("legacy control is single-stage only; use the default async controller for the full game")
    if args.control == "async":
        from async_runner import run as run_async
        run_async(args)
    else:
        run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
