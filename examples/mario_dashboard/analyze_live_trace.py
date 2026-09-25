"""Summarize sampled model authority and frame-accurate failure windows honestly."""
import argparse
from collections import Counter
import json
from pathlib import Path

def analyze(path, world=1, stage=3):
    groups={}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            try: row=json.loads(line)
            except json.JSONDecodeError: continue  # an actively appended final line
            session=row.get('session',{})
            if (session.get('world'),session.get('stage'))!=(world,stage): continue
            run=session.get('run_id',0)
            group=groups.setdefault(run,{'sample_count':0,'sampled_model_goals':Counter(),
                'accepted_samples':0,'model_changed_action_samples':0,'control_sources':Counter(),
                'model_ages_frames':[],'model_snapshots':set(),'deaths':[]})
            if row.get('type')=='sample':
                d=row.get('decision',{});group['sample_count']+=1
                group['sampled_model_goals'][str(d.get('model_goal'))]+=1
                group['control_sources'][str(d.get('source'))]+=1
                group['accepted_samples']+=bool(d.get('intent_accepted'))
                group['model_changed_action_samples']+=bool(d.get('model_changed_action'))
                if d.get('source_snapshot'):group['model_snapshots'].add(d['source_snapshot'])
                if row.get('decision_age_frames') is not None:group['model_ages_frames'].append(row['decision_age_frames'])
            if row.get('type')=='lifecycle' and row.get('event')=='life_lost':
                trace=row.get('recent_trace',[])
                group['deaths'].append({'session_frame':session.get('session_frame'),
                    'lives_remaining':session.get('lives_remaining'),'trace_frames':len(trace),
                    'last_pre_action':trace[-1] if trace else None,
                    'warning':None if trace else 'Old log has only ~1 Hz samples; exact failure attribution unavailable'})
    for group in groups.values():
        group['unique_sampled_model_snapshots']=len(group.pop('model_snapshots'))
    return {'source':str(path),'world':world,'stage':stage,'runs':groups,
            'limits':'Sample counts are not all game frames or all inference responses. Zero sampled motor changes does not prove zero changes between samples. Post-step event_info may already be a respawn.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trace',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=analyze(args.trace)
    text=json.dumps(result,indent=2)
    if args.output:args.output.write_text(text,encoding='utf-8')
    print(text)
