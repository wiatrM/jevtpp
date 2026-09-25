#!/usr/bin/env python3
"""Actual full-ROM lifecycle smoke: continue through a flag into the next stage.

Uses no model, save-state, position script, or framebuffer shortcut. This is a
bounded transition regression, not a claim that the controller clears 8 worlds.
"""
from __future__ import annotations
import argparse
import json
from collections import deque
from pathlib import Path
from async_runner import FastControllerProcess
from game_session import GameSession
from nes_state import StateTracker, unwrap_ram
from run_nes_demo import (create_environment, reset_environment, step_environment,
                          current_environment_info, ACTION_INDEX)


def smoke(executable, max_frames=4000, stop_stage=2, screenshot=None):
    env=create_environment("SuperMarioBros-v0")
    control=FastControllerProcess(executable,environment="SuperMarioBros-v0")
    tracker=StateTracker(1)
    _,initial=reset_environment(env,0)
    info=current_environment_info(env,initial)
    session=GameSession(); session.start(info)
    epoch,frame,previous,reward=1,0,"noop",0
    events=[]; outcome="frame_budget"
    recent=deque(maxlen=120)
    try:
        for total in range(max_frames):
            state,_=tracker.parse(info,unwrap_ram(env),previous_action=previous,
                                  previous_reward=reward,response_delay_frames=0)
            state["session"]=session.snapshot(epoch)
            decision=control.step(state,{},epoch,frame,reset=frame==0)
            recent.append({"frame":total,"player":state["player"],"hazard":state["hazard"],
                           "collision":state["collision"],"action":decision["action"],
                           "reason":decision.get("reason"),"phase":decision.get("phase"),
                           "risk":decision.get("risk_budget"),"outcome":decision.get("outcome_update")})
            previous=decision["action"]
            observation,reward,terminated,truncated,event_info=step_environment(env,ACTION_INDEX[previous])
            info=current_environment_info(env,event_info)
            outcome=session.advance(event_info,info,terminated=terminated,truncated=truncated)
            frame+=1
            if session.stage_transition or session.life_lost:
                events.append({"frame":total+1,"event_info":event_info,"current_info":info,
                               "session":session.snapshot(epoch),"recent":list(recent) if session.life_lost else []})
                recent.clear()
                epoch+=1; frame=0; tracker.reset(); previous="noop"
            if outcome: break
            if (session.world,session.stage)>=(1,stop_stage) and session.stage_frame>=120:
                outcome=f"stage_{session.world}_{session.stage}_playing"; break
        else:
            outcome="frame_budget"
        if screenshot is not None:
            from PIL import Image
            Image.fromarray(observation).save(screenshot)
        state,_=tracker.parse(info,unwrap_ram(env),previous_action=previous,
                              previous_reward=reward,response_delay_frames=0)
        state["session"]=session.snapshot(epoch)
        final=control.step(state,{},epoch,frame,include_memory=True)
        knowledge=final.get("knowledge",{})
        map_levels=sorted({(item["world"],item["stage"]) for item in knowledge.get("columns",[])})
        return {"environment":"SuperMarioBros-v0","outcome":outcome,"frames":total+1,
                "session":session.snapshot(epoch),"final_info":info,"events":events,
                "tail":list(recent),
                "map_levels":map_levels,"observed_columns":len(knowledge.get("columns",[])),
                "memory_outcomes":knowledge.get("outcomes",{}),
                "provenance":"Actual full-game NES environment, C++ online planner with reactive fallback; no model or replay.",
                "limitation":(f"Stops after 120 active frames of stage1-{stop_stage}; does not validate full-game completion."
                              if stop_stage<=4 else "Runs until terminal or frame budget; report actual outcome, not an assumed win.")}
    finally:
        control.close(); env.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--controller-jsonl",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-frames",type=int,default=4000)
    parser.add_argument("--stop-stage",type=int,default=2)
    parser.add_argument("--screenshot",type=Path)
    args=parser.parse_args()
    result=smoke(args.controller_jsonl,args.max_frames,args.stop_stage,args.screenshot)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ("events","tail")}),flush=True)
