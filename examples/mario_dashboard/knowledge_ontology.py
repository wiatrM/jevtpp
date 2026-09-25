"""Bounded, evidence-linked ontology projection for the Mario dashboard.

The C++ controller owns actions and learning. This view only joins its observed
telemetry; a model preference changes salience, never the truth of an edge.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from nes_state import ENEMY_NAMES


class KnowledgeOntology:
    def __init__(self, retain_across_runs=False):
        self.retain_across_runs = retain_across_runs
        self.reset()

    def reset(self):
        self.scope = None
        self.nodes = OrderedDict()
        self.edges = OrderedDict()
        self.previous = None
        self.last_flight = None
        self.last_run = None
        self.runs_observed = set()
        self.jump_variants = {}
        self.death_zones = {}
        self.generation = getattr(self, "generation", 0) + 1

    def set_retention(self, enabled):
        self.retain_across_runs = bool(enabled)
        if enabled and self.scope and self.last_run is not None:
            # Promoting the current attempt must not erase observations that
            # were already gathered before the operator flipped the switch.
            self.scope=("session",self.scope[-1])

    @staticmethod
    def _enemy_name(kind):
        try:
            return ENEMY_NAMES.get(int(kind), f"unknown_enemy_{kind}").replace("_", " ").upper()
        except (TypeError, ValueError):
            return f"UNKNOWN ENEMY {kind}"

    def _node(self, ident, kind, label, frame, *, status="observed", source="RAM",
              attributes=None, salience=0):
        item = self.nodes.get(ident, {"id": ident, "first_frame": frame})
        item.update(kind=kind, label=label, status=status, source=source,
                    last_frame=frame, attributes=attributes or {}, salience=salience)
        self.nodes[ident] = item
        self.nodes.move_to_end(ident)
        while len(self.nodes) > 200:
            victim=next((key for key,node in self.nodes.items()
                         if node.get("kind")=="enemy"),next(iter(self.nodes)))
            self.nodes.pop(victim)

    def _edge(self, src, dst, relation, frame, *, status="observed", source="RAM",
              attributes=None, increment=False, salience=0):
        ident=f"{src}|{relation}|{dst}"
        item=self.edges.get(ident, {"id":ident,"from":src,"to":dst,"relation":relation,
                                   "first_frame":frame,"events":0})
        item.update(status=status,source=source,last_frame=frame,
                    attributes=attributes or {},salience=salience)
        if increment:
            item["events"]+=1
        self.edges[ident]=item
        self.edges.move_to_end(ident)
        while len(self.edges)>350:
            self.edges.popitem(last=False)

    def observe_outcome(self, *, state, control, session, frame, run_id, recent_trace=()):
        """Join post-step life loss to the last pre-step observation."""
        if not session.get("life_lost"):
            return self.snapshot(frame)
        level=state.get("level",{})
        stage=f"{level.get('world','?')}-{level.get('stage','?')}"
        player=state.get("player",{})
        x=player.get("x")
        if not isinstance(x,(int,float)) or not math.isfinite(x):
            return self.snapshot(frame)
        bucket=int(x//32)*32
        ident=f"hazard:{stage}:{bucket}"
        record=self.death_zones.setdefault(ident,{"stage":stage,"x":bucket+16,
                                                   "events":0,"runs":set(),"actions":{},
                                                   "unsafe_intercepts":0,"unsafe_waits":0,
                                                   "target_kinds":set()})
        record["events"]+=1
        record["runs"].add(run_id)
        action=control.get("action","unknown")
        record["actions"][action]=record["actions"].get(action,0)+1
        target=control.get("persistent_tasks",{}).get("target")
        if not isinstance(target,dict) or not target.get("visible"):
            # A target may become unobserved on the collision/death frame.
            # Recover only a *recently visible* target; do not mark its
            # disappearance as a kill or infer a collision cause.
            target=next((step.get("task",{}).get("target") for step in
                         reversed(list(recent_trace)[-8:])
                         if isinstance(step.get("task",{}).get("target"),dict)
                         and step["task"]["target"].get("visible")),None)
        near_target=isinstance(target,dict) and target.get("visible") and isinstance(target.get("x"),(int,float)) and abs(target["x"]-x)<=24
        # Credit the preceding *sequence*, not the final input after the
        # emulator has already entered its death animation.
        coast=streak=wait_frames=0
        for step in list(recent_trace)[-48:]:
            if (near_target and step.get("reason")=="hunter_track_intercept" and
                    step.get("action")=="noop" and
                    step.get("task",{}).get("target",{}).get("kind_id")==target.get("kind_id")):
                streak+=1
                coast=max(coast,streak)
            else:
                streak=0
            if (near_target and step.get("reason")=="hunter_wait_for_reachable_target" and
                    step.get("task",{}).get("target",{}).get("kind_id")==target.get("kind_id")):
                wait_frames+=1
        unsafe_intercept=near_target and coast>=8
        unsafe_wait=near_target and wait_frames>=12
        if unsafe_intercept:
            record["unsafe_intercepts"]+=1
        if unsafe_wait:
            record["unsafe_waits"]+=1
        if (unsafe_intercept or unsafe_wait) and isinstance(target.get("kind_id"),int):
            record["target_kinds"].add(target["kind_id"])
        self._node(ident,"danger_zone",f"{stage} · X {bucket}–{bucket+31}",frame,
                   status="confirmed",source="post-step life counter transition",
                   attributes={"stage":stage,"x":record["x"],"events":record["events"],
                               "runs":len(record["runs"]),"last_action":action,
                               "unsafe_intercepts":record["unsafe_intercepts"],
                               "unsafe_waits":record["unsafe_waits"],
                               "preceding_coast_frames":coast},salience=1)
        self._node("outcome:life_lost","outcome","LIFE LOST",frame,status="confirmed",
                   source="post-step life counter transition",salience=1)
        self._edge(ident,"outcome:life_lost","observed_life_loss_near",frame,
                   status="confirmed",source="post-step life counter + pre-step position",
                   increment=True,attributes={"stage":stage,"x":x,"run_id":run_id,
                                              "action":action,"events":record["events"]},salience=1)
        if near_target:
            enemy=f"enemy_type:{target.get('kind_id','?')}"
            self._node(enemy,"enemy_type",self._enemy_name(target.get("kind_id")),frame,
                       status="observed",source="pre-step RAM target",salience=.8)
            self._edge(ident,enemy,"nearby_at_life_loss",frame,status="associated",
                       source="pre-step RAM proximity; cause unconfirmed",increment=True,
                       attributes={"distance_px":round(target["x"]-x,1),"run_id":run_id},salience=.9)
            if unsafe_intercept:
                self._node("skill:track_intercept","skill","AIRBORNE STOMP INTERCEPT",
                           frame,status="observed",source="executed C++ controller action",salience=.9)
                self._edge("skill:track_intercept",ident,"observed_failure_context",frame,
                           status="empirical",source="executed skill + post-step life loss",
                           increment=True,attributes={"enemy_kind_id":target.get("kind_id"),
                                                      "preceding_coast_frames":coast,
                                                      "action":action,"run_id":run_id},salience=1)
            if unsafe_wait:
                self._node("skill:wait_for_reachable_target","skill","WAIT BELOW HIGH TARGET",
                           frame,status="observed",source="executed C++ controller action",salience=.9)
                self._edge("skill:wait_for_reachable_target",ident,"observed_failure_context",frame,
                           status="empirical",source="executed waiting + post-step life loss",
                           increment=True,attributes={"enemy_kind_id":target.get("kind_id"),
                                                      "preceding_wait_frames":wait_frames,
                                                      "run_id":run_id},salience=1)
        return self.snapshot(frame)

    def observe(self, state, control, knowledge, model, *, frame, run_id, memory_generation):
        scope=("session",memory_generation) if self.retain_across_runs else (run_id,memory_generation)
        if self.scope!=scope:
            self.reset()
            self.scope=scope
        if self.last_run is not None and self.last_run!=run_id:
            # Entity IDs, flight signatures and adjacent transitions are not
            # transferable between physical runs, even when evidence is.
            self.previous = None
            self.last_flight = None
        self.last_run = run_id
        self.runs_observed.add(run_id)
        session=state.get("session", {})
        if session.get("game_over"):
            if not self.retain_across_runs:
                self.reset()
                self.scope=scope
            return self.snapshot(frame)
        player=state.get("player", {})
        level=state.get("level", {})
        detectors=state.get("detectors", {})
        context=knowledge.get("context", {})
        graph={n["id"]:n.get("output",{}) for n in control.get("graph",{}).get("nodes",[])}
        dynamics=graph.get("action_dynamics",{})
        belief=graph.get("world_belief",{})
        phase=f"{level.get('world','?')}-{level.get('stage','?')}"
        size=player.get("powerup_status","unknown")
        self._node("mario","agent","MARIO",frame,attributes={"x":player.get("x"),"size":size},salience=1)
        self._node(f"stage:{phase}","place",f"WORLD {phase}",frame,attributes={
            "known_columns":len(knowledge.get("columns",[]))},salience=.7)
        self._edge("mario",f"stage:{phase}","currently_in",frame,salience=.7)
        self._node(f"size:{size}","state",size.upper(),frame,salience=.5)
        self._edge("mario",f"size:{size}","observed_size",frame,salience=.5)
        self._node("jump","action","HOLD A / JUMP",frame,status="empirical",
                   source="confirmed flight telemetry",salience=.8)
        n=context.get("jump_samples",0)
        if n:
            duration=context.get("jump_duration_mean_frames")
            distance=context.get("jump_range_mean_pixels")
            self._node("jump:airtime","effect","AIR TIME",frame,status="empirical",
                       source="confirmed jumps",attributes={"mean_frames":duration,"samples":n},salience=.7)
            self._node("jump:range","effect","HORIZONTAL RANGE",frame,status="empirical",
                       source="confirmed jumps",attributes={"mean_pixels":distance,"samples":n},salience=.7)
            self._edge("jump","jump:airtime","observed_duration",frame,status="empirical",
                       source="confirmed jumps",attributes={"samples":n,"mean_frames":duration},salience=.7)
            self._edge("jump","jump:range","observed_range",frame,status="empirical",
                       source="confirmed jumps",attributes={"samples":n,"mean_pixels":distance},salience=.7)
        flight=dynamics.get("last_flight") or {}
        flight_key=(flight.get("start_frame"),flight.get("end_frame"),flight.get("signature"))
        if flight.get("takeoff_confirmed") and flight.get("end_frame") is not None and flight_key!=self.last_flight:
            self.last_flight=flight_key
            hold=flight.get("held_frames",0)
            bucket=f"{min(24,hold//8*8)}+" if hold>=24 else f"{hold//8*8}-{hold//8*8+7}"
            action=f"jump:hold:{bucket}"
            self._node(action,"action_variant",f"HOLD A {bucket} FR",frame,status="empirical",
                       source="observed input",salience=.8)
            result=flight.get("result","unknown")
            outcome=f"flight:{result}"
            distance=abs((flight.get("landed_x") or flight.get("last_x") or 0)-
                         (flight.get("start_x") or 0))
            self._node(outcome,"outcome",result.replace("_"," ").upper(),frame,
                       status="confirmed" if result=="landing_confirmed" else "uncertain",
                       source="flight verifier",salience=.7)
            self._edge("jump",action,"duration_variant",frame,status="empirical",
                       source="observed input",increment=True,salience=.8)
            self._edge(action,outcome,"observed_flight",frame,
                       status="confirmed" if result=="landing_confirmed" else "uncertain",
                       source="flight verifier",increment=True,attributes={
                           "held_frames":hold,"travel_pixels":round(distance,1),
                           "duration_frames":flight["end_frame"]-flight.get("start_frame",flight["end_frame"]),
                           "contact_suspected":flight.get("contact_suspected",False)},salience=.8)
            start_y,apex_y=flight.get("start_feet_y"),flight.get("apex_feet_y")
            if (result=="landing_confirmed" and not flight.get("contact_suspected") and
                    isinstance(start_y,(int,float)) and isinstance(apex_y,(int,float)) and
                    0 < start_y-apex_y < 240):
                height=start_y-apex_y
                samples=self.jump_variants.setdefault(bucket,[])
                samples.append(height)
                if len(samples)>64:
                    samples.pop(0)
                effect=f"jump:height:{bucket}"
                self._node(effect,"effect",f"APEX {sum(samples)/len(samples):.0f} PX",frame,
                           status="empirical",source="confirmed contact-free flights",
                           attributes={"mean_height_px":round(sum(samples)/len(samples),1),
                                       "samples":len(samples),"hold_bucket":bucket},salience=.75)
                self._edge(action,effect,"observed_apex_height",frame,status="empirical",
                           source="confirmed contact-free flights",increment=True,
                           attributes={"last_height_px":round(height,1),"samples":len(samples)},salience=.75)
            target=flight.get("target")
            if (result=="landing_confirmed" and not flight.get("contact_suspected") and
                    isinstance(target,dict) and isinstance(start_y,(int,float)) and
                    isinstance(target.get("y"),(int,float)) and start_y-target["y"]>=16):
                stage_id=f"stage:{phase}"
                self._edge(action,stage_id,"reached_higher_support",frame,status="empirical",
                           source="confirmed landing + observed support geometry",increment=True,
                           attributes={"rise_px":round(start_y-target["y"],1)},salience=.8)
        speed=context.get("enemy_speeds",{})
        for kind,data in list(speed.items())[:8]:
            if not isinstance(data,dict) or not data.get("samples"):
                continue
            ident=f"enemy_type:{kind}"
            velocity=data.get("mean")
            if not isinstance(velocity,(float,int)) or not math.isfinite(velocity):
                continue
            self._node(ident,"enemy_type",self._enemy_name(kind),frame,status="empirical",
                       source="RAM motion samples",attributes={"samples":data["samples"]},salience=.6)
            motion=f"motion:{kind}"
            self._node(motion,"motion",f"{velocity:+.2f} PX/FR",frame,status="empirical",
                       source="RAM motion samples",attributes={"mean_vx":velocity,"samples":data["samples"]},salience=.6)
            self._edge(ident,motion,"observed_horizontal_motion",frame,status="empirical",
                       source="RAM motion samples",attributes={"mean_vx":velocity,"samples":data["samples"]},salience=.6)
        for enemy in (belief.get("entities") or [])[-8:]:
            if not isinstance(enemy,dict) or "id" not in enemy:
                continue
            entity=f"enemy:{run_id}:{enemy['id']}"
            kind=f"enemy_type:{enemy.get('kind_id','?')}"
            status=enemy.get("status","unresolved")
            self._node(entity,"enemy",f"{self._enemy_name(enemy.get('kind_id'))} #{enemy['id']} · R{run_id}",frame,
                       status=status,source="RAM identity tracker",
                       attributes={"x":enemy.get("x"),"vx":enemy.get("vx"),
                                   "visible":enemy.get("visible"),"kind_id":enemy.get("kind_id")},
                       salience=.95 if status=="observed" else .3)
            self._node(kind,"enemy_type",self._enemy_name(enemy.get('kind_id')),frame,
                       status="observed",source="RAM type",salience=.6)
            self._edge(entity,kind,"instance_of",frame,status=status,
                       source="RAM identity tracker",salience=.65)
        prior=self.previous
        collected=detectors.get("powerups_collected",0)
        if (prior and prior["run_id"]==run_id and prior["stage"]==phase and
                collected>prior["collected"] and prior["size"]=="small" and size in {"tall","big"}):
            self._node("powerup:mushroom","powerup","MUSHROOM / GROWTH",frame,status="confirmed",
                       source="pickup counter + size transition",salience=.9)
            self._edge("powerup:mushroom",f"size:{size}","observed_growth",frame,status="confirmed",
                       source="pickup counter + RAM size transition",increment=True,salience=.9)
        if (prior and prior["run_id"]==run_id and prior["stage"]==phase and
                prior["size"] in {"tall","big"} and size=="small" and
                session.get("lives_remaining")==prior["lives"]):
            self._edge(f"size:{prior['size']}","size:small","observed_size_loss_without_life_loss",frame,
                       status="confirmed",source="RAM size + lives transition",
                       increment=True,salience=.9)
        star_timer=player.get("star_timer_ram_0x079f")
        if isinstance(star_timer,int) and star_timer>0:
            self._node("effect:star","effect","STAR INVINCIBILITY",frame,
                       status="observed",source="NES RAM $079F timer",
                       attributes={"timer":star_timer,"active":True},salience=.95)
            self._edge("mario","effect:star","active_star_timer",frame,
                       status="observed",source="NES RAM $079F timer",salience=.95)
            if prior and prior["run_id"]==run_id and prior.get("star_timer")==0:
                self._node("powerup:star","powerup","STAR POWER-UP",frame,
                           status="observed",source="RAM star timer activation",salience=.9)
                self._edge("powerup:star","effect:star","activates_invincibility_timer",frame,
                           status="observed",source="RAM $079F transition 0→positive",
                           increment=True,salience=.9)
        elif star_timer==0 and "effect:star" in self.nodes:
            self._node("effect:star","effect","STAR INVINCIBILITY",frame,
                       status="observed",source="NES RAM $079F timer",
                       attributes={"timer":0,"active":False},salience=.3)
        # This is an event association, not a learned causal law. The RAM
        # defeated transition, spatial overlap, no stomp and unchanged lives
        # must all agree; disappearance or a score increase alone is excluded.
        if (prior and prior["run_id"]==run_id and isinstance(star_timer,int) and
                star_timer>0 and not session.get("life_lost") and
                session.get("lives_remaining") is not None and
                session.get("lives_remaining")==prior["lives"] and
                detectors.get("stomps_confirmed",0)==prior.get("stomps",0)):
            for event in state.get("hazard",{}).get("elimination_events",[]):
                if not isinstance(event,dict) or event.get("source")!="RAM_alive_to_defeated":
                    continue
                ex,ey=event.get("x"),event.get("screen_y")
                px,py=player.get("x"),player.get("screen_y")
                if not all(isinstance(v,(int,float)) for v in (ex,ey,px,py)):
                    continue
                if abs(ex-px)>24 or abs(ey-py)>40:
                    continue
                kind=event.get("kind_id")
                if not isinstance(kind,int):
                    continue
                enemy_type=f"enemy_type:{kind}"
                self._node(enemy_type,"enemy_type",self._enemy_name(kind),frame,
                           status="observed",source="RAM defeated transition",salience=.8)
                self._edge("effect:star",enemy_type,
                           "contact_elimination_without_life_loss_while_star_active",frame,
                           status="empirical",source="RAM star timer + enemy defeated + overlap + unchanged lives; no stomp",
                           increment=True,attributes={"enemy_x":ex,"mario_x":px,
                           "star_timer":star_timer,"run_id":run_id,"frame":frame},salience=1)
        self.previous={"run_id":run_id,"stage":phase,"size":size,"collected":collected,
                       "lives":session.get("lives_remaining"),"star_timer":star_timer,
                       "stomps":detectors.get("stomps_confirmed",0)}
        # The last model answer may highlight existing knowledge. It adds no facts.
        current_judgment=control.get("plan_intent",{}).get("accepted",False)
        if current_judgment and model.get("judgment_kind")=="plan" and model.get("selected_slot")!=6:
            selected=model.get("model_plan","")
            if "jump" in selected or "traverse" in selected:
                focus=["jump","jump:range","jump:airtime"]
            elif "stomp" in selected:
                focus=[n for n in self.nodes if n.startswith(f"enemy:{run_id}:")][-2:]
            else:
                focus=[]
        else:
            focus=[]
        return self.snapshot(frame,focus)

    def snapshot(self,frame,focus=()):
        nodes=list(self.nodes.values())
        edges=[edge for edge in self.edges.values() if edge["from"] in self.nodes and edge["to"] in self.nodes]
        for node in nodes:
            node["model_focus"]=node["id"] in focus
        return {"schema":"jevt.knowledge_view.v1","scope":"cross_run_session" if self.retain_across_runs else "current_game_run",
                "generation":self.generation,"frame":frame,"nodes":nodes,"edges":edges,
                "runs_observed":len(self.runs_observed),
                "model_focus":list(focus),"model_focus_policy":"Only a current accepted plan judgment can highlight facts; highlighting cannot assert a relation",
                "provenance":"RAM and verified transition projection; model chooses salience only",
                "limits":"Observed correlations are not learned causal laws; adjacent frame samples are dependent. Unknown remains unknown."}

    def model_context(self, *, run_id, stage, mode, star_active=False, limit=12):
        """Bounded, provenance-bearing relation language for the plan ranker."""
        prefix=f"enemy:{run_id}:"
        priorities={"observed_failure_context":10,"observed_life_loss_near":9,"nearby_at_life_loss":8,
                    "active_star_timer":5,"activates_invincibility_timer":5,
                    "contact_elimination_without_life_loss_while_star_active":5,
                    "observed_apex_height":4,"reached_higher_support":4,
                    "observed_horizontal_motion":3,"instance_of":3,
                    "observed_growth":2,"observed_size_loss_without_life_loss":2}
        current=f"stage:{stage}"
        edges=sorted(self.edges.values(),key=lambda e:(
            -(priorities.get(e["relation"],1)+(3 if e["from"].startswith(prefix) else 0)+
              (2 if current in (e["from"],e["to"]) else 0)), -e["last_frame"]))
        statements=[]
        relation_counts={}
        for edge in edges:
            source=self.nodes.get(edge["from"]);target=self.nodes.get(edge["to"])
            if not source or not target: continue
            if source["kind"]=="danger_zone" and source.get("attributes",{}).get("stage")!=stage: continue
            if source["kind"]=="enemy" and not edge["from"].startswith(prefix): continue
            if source["kind"]=="enemy" and source["status"]!="observed": continue
            if edge["relation"]=="active_star_timer" and not star_active: continue
            if edge["relation"]=="currently_in" and edge["to"]!=current: continue
            if relation_counts.get(edge["relation"],0)>=2: continue
            statements.append({"subject":source["label"],"relation":edge["relation"],
                               "object":target["label"],"status":edge["status"],
                               "source":edge["source"],"samples":edge.get("events") or edge.get("attributes",{}).get("samples",0),
                               "attributes":edge.get("attributes",{})})
            relation_counts[edge["relation"]]=relation_counts.get(edge["relation"],0)+1
            if len(statements)>=limit: break
        return {"schema":"jevt.knowledge_context.v1","mode":mode,"stage":stage,
                "scope":"cross_run_session" if self.retain_across_runs else "current_game_run",
                "statements":statements,
                "interpretation":"Death location is confirmed; nearby enemy is only an association. Unknown terrain and unseen enemies remain unknown."}

    def affordances(self):
        """Verified repeated event associations, never an unchecked model claim."""
        relation="contact_elimination_without_life_loss_while_star_active"
        kinds=[]
        for edge in self.edges.values():
            if edge["relation"]!=relation or edge["events"]<2 or edge["status"]!="empirical":
                continue
            target=edge["to"]
            if target.startswith("enemy_type:"):
                try:
                    kinds.append(int(target.split(":",1)[1]))
                except ValueError:
                    pass
        zones=[{"stage":zone["stage"],"x":zone["x"],"events":zone["events"],
                "runs":len(zone["runs"]),"warning_radius_px":112,
                "unsafe_intercepts":zone["unsafe_intercepts"],
                "unsafe_waits":zone["unsafe_waits"],
                "target_kinds":sorted(zone["target_kinds"])}
               for zone in self.death_zones.values()]
        return {"schema":"jevt.mario_affordance.v1","star_contact_kinds":sorted(set(kinds))[:8],
                "death_zones":zones[-24:],
                "source_relation":relation,"minimum_confirmed_events":2,
                "scope":"observed_association_not_general_rule"}
