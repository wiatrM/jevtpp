"""Two-rate NES loop: frame-accurate JevT++ control, asynchronous model intent."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import selectors
import subprocess
import threading
import tempfile
import time
import urllib.request

from nes_state import StateTracker, unwrap_ram
from game_session import GameSession
from model_selection import ModelSelection, laya_runtime
from judgment_contract import bind_plan_judgment
from knowledge_ontology import KnowledgeOntology
from run_nes_demo import (ACTIONS, ACTION_INDEX, DashboardHandler, LiveState, ThreadingHTTPServer,
                         call_model, create_environment, current_environment_info, encode_frame, reset_environment,
                         start_engine, step_environment)

def attach_knowledge_context(request, graph_context):
    """Keep a short evidenced prefix visible to bounded-context models."""
    head=[{key:fact.get(key) for key in
           ("subject","relation","object","status","samples")}
          for fact in graph_context.get("statements",[])[:3]]
    return {"knowledge_head":head,**request,"knowledge_graph":graph_context}


class FastControllerProcess:
    """Persistent JSONL C++ graph, not a Python replacement for JevT++."""
    def __init__(self, executable, memory_path=None, environment="SuperMarioBros-1-1-v0"):
        self.process = subprocess.Popen([str(Path(executable).resolve()), "--env", environment], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.memory_path = Path(memory_path) if memory_path else None
        if self.memory_path and self.memory_path.exists():
            try:
                saved = json.loads(self.memory_path.read_text(encoding="utf-8"))
                result = self.exchange({"command": "memory_import", "memory": saved})
                if not result.get("ok"):
                    print("Knowledge memory rejected: " + result.get("error", "invalid"), flush=True)
            except (OSError, ValueError) as error:
                print(f"Knowledge memory not loaded: {error}", flush=True)

    def exchange(self, request):
        self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        if not self.selector.select(timeout=3):
            raise TimeoutError("JevT++ control graph exceeded 3 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("JevT++ control graph stopped")
        return json.loads(line)

    def step(self, state, strategy, epoch, frame, reset=False, include_memory=False):
        request = {"state": state, "strategy": strategy, "epoch": epoch, "frame": frame, "reset": reset,
                   "include_memory": include_memory}
        result = self.exchange(request)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "control graph failed"))
        if result.get("action") not in ACTION_INDEX:
            raise RuntimeError("C++ graph produced an unknown controller action")
        return result

    def save_memory(self):
        if self.memory_path is None:
            return
        result = self.exchange({"command": "memory_export"})
        if not result.get("ok"):
            raise RuntimeError("Unable to export knowledge memory")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.memory_path.parent,
                                         prefix=self.memory_path.name + ".", suffix=".tmp", delete=False) as file:
            json.dump(result["memory"], file, separators=(",", ":"))
            temporary = Path(file.name)
        os.replace(temporary, self.memory_path)

    def close(self):
        self.selector.close()
        self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class StrategyWorker:
    """At most one in-flight request. No growing queue of obsolete observations."""
    def __init__(self, url):
        self.url = url
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jevt-strategy")
        self.future = None
        self.latest = {}
        self.error = None
        self.completed = self.discarded = 0
        self.last_submit_frame = -10000
        self.pending_epoch = None
        self.pending_mode = None
        self.generation = self.pending_generation = 0

    def switch_url(self, url):
        self.url=url
        self.generation+=1
        self.reset()

    def reset(self):
        self.latest = {}
        self.error = None
        self.last_submit_frame = -10000
        # Cannot cancel running HTTP; epoch checks discard its eventual result.

    def submit(self, compact, epoch, frame, mode):
        if self.future is not None:
            return False
        plan_request='plan_candidates' in compact
        if plan_request and not compact['plan_candidates']:
            self.last_submit_frame=frame
            return False
        # The producer may reuse/mutate nested dictionaries after submit. Both
        # inference and slot binding must see this exact immutable-by-ownership
        # snapshot, not a later context relabeled with an older frame number.
        compact=deepcopy(compact)
        url=self.url.replace('/api/strategy','/api/plan') if plan_request else self.url
        def request():
            started = time.perf_counter()
            result = call_model(url, compact)
            if not result.get("ok"):
                raise RuntimeError(result.get("error", "model evaluation failed"))
            if plan_request:
                result=bind_plan_judgment(result,compact)
            return {**result, "epoch": epoch, "source_frame": frame, "mode": mode,
                    "source_snapshot": f"{epoch}:{frame}",
                    "wall_ms": (time.perf_counter() - started) * 1000}
        self.future = self.pool.submit(request)
        self.pending_epoch, self.pending_mode = epoch, mode
        self.pending_generation=self.generation
        self.last_submit_frame = frame
        return True

    def poll(self, epoch, mode):
        if self.future is None or not self.future.done():
            return False
        if self.pending_epoch != epoch or self.pending_mode != mode or self.pending_generation != self.generation:
            self.discarded += 1
            self.future = None
            return True
        try:
            result = self.future.result()
            if result["epoch"] == epoch and result["mode"] == mode:
                self.latest = result
                self.error = None
                self.completed += 1
            else:
                self.discarded += 1
        except Exception as error:
            self.error = str(error)
        finally:
            self.future = None
        return True

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def compose_decision(model, control, control_ms):
    # Large trace and map objects live once at the top level of the stream.
    fields = {key: value for key, value in control.items()
              if key not in {"graph", "knowledge", "compact_state", "model_request", "digital_twin",
                             "game_phase", "learning_delta", "risk_policy", "outcome_update",
                             "world_belief", "action_dynamics", "persistent_tasks", "risk_budget"}}
    return {**model, **fields,
            "plan_intent_accepted":control.get('plan_intent',{}).get('accepted',False),
            "plan_model_applied":bool(control.get('planner_applied',False) and control.get('risk_budget',{}).get('model_rank_applied',False)),
            "backend": model.get("backend", "Open-JEV pending · JevT++ reflex"),
            "model_goal": model.get("active_goal"), "latency_ms": model.get("wall_ms"),
            "control_ms": control_ms, "gate_reason": control.get("reason", ""),
            "active_goal": control.get("active_goal", model.get("active_goal", ""))}


def run(args):
    base = Path(__file__).resolve().parent
    controller_path = Path(args.controller or Path(args.engine).with_name("jevt_mario_control"))
    if not controller_path.is_file():
        raise RuntimeError("Build jevt_mario_control before starting the asynchronous demo")
    runtime = {}
    if args.open_jev_url:
        with urllib.request.urlopen(args.open_jev_url.rstrip("/") + "/api/runtime", timeout=5) as response:
            service = json.load(response)
            runtime = {**service.get("runtime", {}),
                       "base_model_id": service.get("model"),
                       "open_jev_method": service.get("method")}
        if (runtime.get("validated") is not True or not runtime.get("cuda_available") or
                not str(runtime.get("device", "")).startswith("cuda:") or
                "4090" not in runtime.get("device_name", "")):
            raise RuntimeError("Open-JEV must verify loaded model on RTX 4090 CUDA; CPU fallback is forbidden")
    engine = start_engine(args, base)
    selected='open_jev' if args.open_jev_url else 'laya' if getattr(args,'model',None) else 'preview'
    if selected=='laya': runtime=laya_runtime(args.model_url)
    runtime['model_kind']=selected
    models=ModelSelection(selected,args.model_url.replace('/api/decision','/api/strategy'),runtime)
    laya_engine=models.add_laya(args,base,start_engine) if selected!='laya' else None
    # Retention is an explicit in-process experiment. It never imports a
    # previous process's save file into the full-game session.
    full_game = args.env in {f"SuperMarioBros-v{version}" for version in range(4)}
    memory_path = None if full_game or getattr(args,'fresh_memory',False) else base / "artifacts" / ("knowledge-" + "".join(c for c in args.env if c.isalnum() or c in "-_") + ".json")
    controller = FastControllerProcess(controller_path, memory_path, args.env)
    worker = StrategyWorker(args.model_url.replace("/api/decision", "/api/strategy"))
    live = LiveState()
    DashboardHandler.live, DashboardHandler.web_root = live, base / "web"
    httpd = ThreadingHTTPServer(("127.0.0.1", args.dashboard_port), DashboardHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="mario-dashboard").start()
    environment = create_environment(args.env)
    tracker = StateTracker(1)
    epoch, frame_index, control_count = 1, 0, 0
    frame, info = reset_environment(environment, args.seed)
    if full_game:
        info = current_environment_info(environment, info)
    session = GameSession()
    session.start(info)
    memory_cleared = False
    action, previous_reward, reward_total = "noop", 0., 0.
    objective_mode, paused, terminal = args.mode, False, False
    reset_control = True
    knowledge = {}
    knowledge_scope=getattr(args,'knowledge_scope','run')
    ontology = KnowledgeOntology(retain_across_runs=knowledge_scope=='session')
    model_knowledge_input=None
    knowledge_actions_run=knowledge_actions_session=0
    next_frame = next_publish = next_log = time.perf_counter()
    run_started = time.perf_counter()
    recent_trace=deque(maxlen=120)
    log_path = base / "artifacts" / f"async-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    print(f"JevT++ async dashboard v16: http://127.0.0.1:{args.dashboard_port}/?dashboard=v16", flush=True)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            manifest={'type':'configuration','version':'v16-knowledge-retention','args':vars(args),
                      'runtime':runtime,'model_selection':models.snapshot(),
                      'learning_scope':knowledge_scope,
                      'controller_binary_sha256':hashlib.sha256(controller_path.read_bytes()).hexdigest()}
            log.write(json.dumps(manifest,default=str,separators=(',',':'))+'\n')
            while True:
                with live.lock:
                    command, live.command = live.command, None
                if command == "pause":
                    paused = True
                    live.update(payload={**live.payload, "status": "paused"})
                elif command == "resume" and not terminal:
                    paused = False
                    next_frame = time.perf_counter()
                if command and command.startswith('model:'):
                    key=command.split(':',1)[1]
                    if key in models.routes and key!=models.selected:
                        url,runtime=models.select(key)
                        epoch+=1
                        worker.switch_url(url)
                        model_knowledge_input=None
                        recent_trace.clear()
                        live.update(payload={**live.payload,'runtime':runtime,'model_selection':models.snapshot(),
                            'epoch':epoch,'decision':{'backend':models.options[key]['label']+' pending · JevT++ controller'},
                            'graph_current':False,'decision_age_frames':None,
                            'timing':{**live.payload.get('timing',{}),'model_wall_ms':None,'model_age_frames':None,'model_error':None}})
                        next_publish=next_log=time.perf_counter()
                        log.write(json.dumps({'type':'model_switch','selected':key,'epoch':epoch,
                                              'frame_index':frame_index,'session':session.snapshot(epoch)})+'\n')
                        log.flush()
                if command and command.startswith('knowledge_scope:'):
                    selected_scope=command.split(':',1)[1]
                    if selected_scope in {'run','session'} and selected_scope!=knowledge_scope:
                        knowledge_scope=selected_scope
                        ontology.set_retention(knowledge_scope=='session')
                        if knowledge_scope=='run':
                            cleared=controller.exchange({'command':'memory_reset'})
                            if not cleared.get('ok'):
                                raise RuntimeError('JevT++ failed to clear knowledge after selecting RUN scope')
                            knowledge=cleared.get('memory',{})
                            session.mark_memory_reset()
                            ontology.reset()
                            reset_control=True
                        epoch+=1
                        worker.reset()
                        live.update(payload={**live.payload,'knowledge_scope':knowledge_scope,
                                             'knowledge':knowledge,'ontology':ontology.snapshot(frame_index),
                                             'session':session.snapshot(epoch),'epoch':epoch})
                        log.write(json.dumps({'type':'knowledge_scope','scope':knowledge_scope,
                                              'epoch':epoch,'run_id':session.run_id})+'\n')
                        log.flush()
                if command == "restart" or (command and command.startswith("mode:")):
                    if command.startswith("mode:"):
                        objective_mode = command.split(":", 1)[1]
                    epoch += 1
                    worker.reset()
                    if command == "restart" or terminal:
                        if full_game and knowledge_scope=='run' and not memory_cleared:
                            cleared = controller.exchange({"command": "memory_reset"})
                            if not cleared.get("ok"):
                                raise RuntimeError("JevT++ failed to reset full-game knowledge")
                            session.mark_memory_reset()
                            knowledge = cleared.get("memory", {})
                        reset_control = True
                        frame, info = reset_environment(environment, args.seed)
                        if full_game:
                            info = current_environment_info(environment, info)
                        session.start(info, new_run=True)
                        knowledge_actions_run=0
                        if knowledge_scope=='run': ontology.reset()
                        memory_cleared = False
                        tracker.reset()
                        recent_trace.clear()
                        model_knowledge_input=None
                        frame_index = control_count = 0
                        reward_total = previous_reward = 0.
                        action = "noop"
                        terminal = paused = False
                        run_started = time.perf_counter()
                        next_publish = next_log = time.perf_counter()
                    next_frame = time.perf_counter()
                model_updated = worker.poll(epoch, objective_mode)
                if model_updated:
                    log.write(json.dumps({'type':'model_response','epoch':epoch,'frame':frame_index,
                        'decision':worker.latest,'error':worker.error,'discarded':worker.discarded},separators=(',',':'))+'\n')
                if paused:
                    if model_updated:
                        with live.lock:
                            settled = {**live.payload,
                                       "timing": {**live.payload.get("timing", {}),
                                                  "in_flight": worker.future is not None,
                                                  "discarded_results": worker.discarded,
                                                  "model_error": worker.error}}
                        live.update(payload=settled)
                    time.sleep(.02)
                    continue
                # Wall-clock inference is not simulated state age. The fast
                # reflex layer sees current RAM; cached intent age is separate.
                state, debug = tracker.parse(info, unwrap_ram(environment), previous_action=action,
                                             previous_reward=previous_reward, response_delay_frames=0,
                                             objective_mode=objective_mode)
                state["session"] = {**session.snapshot(epoch),"retain_knowledge":knowledge_scope=='session'}
                state["knowledge_affordances"]=ontology.affordances()
                control_started = time.perf_counter()
                control = controller.step(state, worker.latest, epoch, frame_index, reset_control,
                                          include_memory=time.perf_counter() >= next_publish)
                knowledge = control.get("knowledge", knowledge)
                ontology_view = ontology.observe(state, control, knowledge, worker.latest,
                    frame=frame_index, run_id=session.run_id,
                    memory_generation=session.memory_generation)
                control_ms = (time.perf_counter() - control_started) * 1000
                if control.get('arbitration',{}).get('selected')=='star_contact':
                    knowledge_actions_run+=1
                    knowledge_actions_session+=1
                    log.write(json.dumps({'type':'knowledge_action','skill':'star_contact',
                        'run_id':session.run_id,'epoch':epoch,'frame':frame_index,
                        'action':control.get('action'),'affordance':state['knowledge_affordances'],
                        'star_timer':state['player'].get('star_timer_ram_0x079f')},
                        separators=(',',':'))+'\n')
                reset_control = False
                action = control["action"]
                if ((args.max_decisions == 0 or worker.completed < args.max_decisions) and
                        frame_index - worker.last_submit_frame >= args.frames_per_decision):
                    request=control.get('model_request',control['compact_state']) if getattr(args,'model_policy','plan')=='plan' else control['compact_state']
                    if getattr(args,'model_policy','plan')=='plan' and request.get('plan_candidates'):
                        graph_context=ontology.model_context(run_id=session.run_id,
                            stage=f"{state['level'].get('world')}-{state['level'].get('stage')}",
                            mode=objective_mode,star_active=state['player'].get('star_active') is True)
                        # LAYA's sequence is capped at 1024 tokens. A short
                        # evidence-bearing signal at the front survives even
                        # when the full canonical graph at the end does not.
                        request=attach_knowledge_context(request,graph_context)
                    if worker.submit(request, epoch, frame_index, objective_mode):
                        model_knowledge_input={"source_snapshot":f"{epoch}:{frame_index}",
                                               "graph":request.get("knowledge_graph"),
                                               "model":models.selected}
                        log.write(json.dumps({'type':'model_request','epoch':epoch,'frame':frame_index,
                            'model':models.selected,'request':request},separators=(',',':'))+'\n')
                model = worker.latest
                age = frame_index - model["source_frame"] if model else None
                decision = compose_decision(model, control, control_ms)
                risk=control.get('risk_budget',{})
                recent_trace.append({'frame':frame_index,'epoch':epoch,'session_frame':session.session_frame,
                    'player':state.get('player',{}),'collision':{k:state.get('collision',{}).get(k) for k in
                        ('planning_columns','landing_surfaces','obstacle_distance_pixels','gap_distance_pixels')},
                    'enemies':state.get('hazard',{}).get('nearby_enemies',[]),
                    'action':action,'reason':control.get('reason'),'source':control.get('source'),
                    'model_goal':model.get('active_goal'),'goal_scores':model.get('goal_probabilities',{}),
                    'model_plan':model.get('model_plan'),'plan_scores':model.get('plan_scores',{}),
                    'plan_intent':control.get('plan_intent'),'arbitration':control.get('arbitration'),
                    'model_snapshot':model.get('source_snapshot'),'model_age_frames':age,
                    'intent_accepted':control.get('intent_accepted'),'model_changed_action':control.get('model_changed_action'),
                    'intent_reason':control.get('intent_reason'),'model_input':control.get('compact_state'),
                    'task':control.get('persistent_tasks'),'flight':control.get('action_dynamics',{}).get('active_flight'),
                    'risk_reason':risk.get('reason'),'selected_plan':risk.get('selected')})
                if getattr(args,'trace_every_frame',False):
                    log.write(json.dumps({'type':'control_frame','session':session.snapshot(epoch),
                        **recent_trace[-1]},separators=(',',':'))+'\n')
                if not model and models.selected in models.options:
                    decision['backend']=models.options[models.selected]['label']+' pending · JevT++ controller'
                payload = {"status": "running", "version": "v16-knowledge-retention", "epoch": epoch,
                           "decision_index": worker.completed, "frame_index": frame_index, "action": action,
                           "decision": decision, "state": state, "debug": debug, "graph": control["graph"],
                           "mode": objective_mode, "knowledge_scope":knowledge_scope,
                           "env": args.env, "episode_reward": reward_total,
                           "session": session.snapshot(epoch),
                           "graph_frame_index": frame_index, "graph_current": True,
                           "runtime": runtime, "knowledge": knowledge,
                           "ontology": ontology_view,
                           "model_knowledge_input":model_knowledge_input,
                           "knowledge_actions_run":knowledge_actions_run,
                           "knowledge_actions_session":knowledge_actions_session,
                           "model_selection":models.snapshot(),
                           "digital_twin": control.get("digital_twin", {}),
                           "game_phase": control.get("game_phase", {}),
                           "learning_delta": control.get("learning_delta", {}),
                           "risk_policy": control.get("risk_policy", {}),
                           "frames_per_decision": args.frames_per_decision, "decision_age_frames": age,
                           "frames_until_decision": max(0, args.frames_per_decision - (frame_index - worker.last_submit_frame)),
                           "timing": {"control_ms": control_ms, "model_wall_ms": model.get("wall_ms"),
                                      "model_age_frames": age, "in_flight": worker.future is not None,
                                      "discarded_results": worker.discarded, "model_error": worker.error,
                                      "emulated_fps": frame_index / max(.001, time.perf_counter() - run_started)}}
                now = time.perf_counter()
                if now >= next_publish:
                    live.update(frame=encode_frame(frame), payload=payload)
                    next_publish = now + 1 / args.dashboard_fps
                if now >= next_log:
                    log.write(json.dumps({"type": "sample", **payload}, separators=(",", ":")) + "\n")
                    log.flush()
                    next_log = now + 1
                frame, reward, terminated, truncated, event_info = step_environment(environment, ACTION_INDEX[action])
                frame_index += 1
                control_count += 1
                previous_reward = reward
                reward_total += reward
                # gym returns event_info before its internal stage/death skip;
                # the returned image and live RAM can already be in the next
                # stage. Keep the event and current observation distinct.
                info = current_environment_info(environment, event_info) if full_game else event_info
                outcome = session.advance(event_info, info, terminated=terminated if full_game else False,
                                          truncated=truncated if full_game else False)
                if full_game and session.life_lost:
                    outcome_view=ontology.observe_outcome(state=state,control=control,
                        session=session.snapshot(epoch),frame=frame_index,run_id=session.run_id,
                        recent_trace=recent_trace)
                    log.write(json.dumps({"type":"knowledge_outcome","event":"life_lost",
                        "epoch":epoch,"frame":frame_index,"run_id":session.run_id,
                        "stage":f"{state['level'].get('world')}-{state['level'].get('stage')}",
                        "x":state["player"].get("x"),"action":action,
                        "death_zones":ontology.affordances()["death_zones"],
                        "graph_nodes":len(outcome_view["nodes"]),"graph_edges":len(outcome_view["edges"])},
                        separators=(",",":"))+"\n")
                if full_game and (session.stage_transition or session.life_lost):
                    log.write(json.dumps({"type": "lifecycle", "event": "stage_transition" if session.stage_transition else "life_lost",
                                          "event_info": event_info, "current_info": info,
                                          "session": session.snapshot(epoch),"recent_trace":list(recent_trace)}, separators=(",", ":")) + "\n")
                    log.flush()
                    recent_trace.clear()
                    if outcome is None:
                        epoch += 1
                        worker.reset()
                        tracker.reset()
                        reset_control = True
                        action, previous_reward = "noop", 0.
                        next_publish = next_log = time.perf_counter()
                limit_reached = args.max_frames > 0 and control_count >= args.max_frames
                stage_terminal = not full_game and (terminated or truncated or bool(event_info.get("flag_get")))
                if outcome in {"game_complete", "game_over", "truncated"} or stage_terminal or limit_reached:
                    final_state, final_debug = tracker.parse(info, unwrap_ram(environment), previous_action=action,
                                                            previous_reward=reward, response_delay_frames=0,
                                                            objective_mode=objective_mode)
                    final_state["session"] = {**session.snapshot(epoch),"retain_knowledge":knowledge_scope=='session'}
                    terminal_started = time.perf_counter()
                    final_control = controller.step(final_state, worker.latest, epoch, frame_index, include_memory=True)
                    terminal_control_ms = (time.perf_counter() - terminal_started) * 1000
                    controller.save_memory()
                    clear = bool(event_info.get("flag_get"))
                    status = outcome if full_game and outcome else (
                        "time_limit" if limit_reached else "clear" if clear else "ended")
                    # Terminal observation is after the action, not its predecessor.
                    terminal_record = {"type": "terminal", "epoch": epoch, "frame_index": frame_index,
                                       "status": status, "flag_get": clear, "terminated": terminated,
                                       "truncated": truncated, "state": final_state,
                                       "event_info": event_info, "info": info,
                                       "session": session.snapshot(epoch),
                                       "recent_trace":list(recent_trace),
                                       "episode_reward": reward_total}
                    log.write(json.dumps(terminal_record, separators=(",", ":")) + "\n")
                    log.flush()
                    if full_game and status == "game_over" and knowledge_scope=='run':
                        cleared = controller.exchange({"command": "memory_reset"})
                        if not cleared.get("ok"):
                            raise RuntimeError("JevT++ failed to clear knowledge on GAME OVER")
                        knowledge = cleared.get("memory", {})
                        session.mark_memory_reset()
                        ontology.reset()
                        memory_cleared = True
                        log.write(json.dumps({"type": "knowledge_reset", "reason": "game_over",
                                              "session": session.snapshot(epoch)}, separators=(",", ":")) + "\n")
                        log.flush()
                    else:
                        knowledge = final_control.get("knowledge", knowledge)
                    live.update(frame=encode_frame(frame), payload={**payload, **terminal_record, "debug": final_debug,
                        "action": final_control["action"], "graph": final_control["graph"],
                        "digital_twin": final_control.get("digital_twin", {}),
                        "game_phase": final_control.get("game_phase", {}),
                        "learning_delta": final_control.get("learning_delta", {}),
                        "risk_policy": final_control.get("risk_policy", {}),
                        "frames_until_decision": 0,
                        "decision": compose_decision(worker.latest, final_control, terminal_control_ms),
                        "ontology": ontology.snapshot(frame_index),
                        "timing": {**payload["timing"], "control_ms": terminal_control_ms,
                                   "model_age_frames": frame_index - model["source_frame"] if model else None},
                        "knowledge": knowledge, "session": session.snapshot(epoch),
                        "graph_frame_index": frame_index, "graph_current": status != "game_over"})
                    terminal = paused = True
                    if args.exit_on_terminal:
                        print(json.dumps({key:value for key,value in terminal_record.items()
                                          if key not in {'recent_trace','state'}}), flush=True)
                        break
                next_frame += 1 / args.fps
                remaining = next_frame - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                elif remaining < -.25:
                    next_frame = time.perf_counter()  # no unbounded catch-up burst
    finally:
        worker.close()
        try:
            controller.save_memory()
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            print(f"Knowledge checkpoint not saved: {error}", flush=True)
        controller.close()
        environment.close()
        httpd.shutdown()
        httpd.server_close()
        for owned_engine in (engine,laya_engine):
            if owned_engine is None: continue
            owned_engine.terminate()
            try:
                owned_engine.wait(timeout=3)
            except subprocess.TimeoutExpired:
                owned_engine.kill()
                owned_engine.wait()
        print(f"Run log: {log_path}", flush=True)
