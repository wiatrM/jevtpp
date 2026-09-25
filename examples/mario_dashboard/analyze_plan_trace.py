"""Descriptive live ranking diagnostics, never a policy-quality comparison."""
import argparse
from collections import Counter
import json
import statistics
from pathlib import Path


def summarize(path):
    counts=Counter(); reasons=Counter(); latencies=[]; decisions=[]; runtime={}
    with Path(path).open() as stream:
        for line in stream:
            try: row=json.loads(line)
            except json.JSONDecodeError: continue  # actively appended final line
            kind=row.get('type'); counts[kind]+=1
            if kind=='configuration':runtime=row.get('runtime',{})
            if kind=='model_response':
                decision=row.get('decision',{})
                key=(decision.get('epoch'),decision.get('source_frame'))
                if decision.get('judgment_kind')=='plan' and key not in decisions:
                    decisions.append(key)
                    counts['unique_plan_responses']+=1
                    counts['model_defer']+=decision.get('selected_slot')==6
                    counts['absent_slot']+=bool(decision.get('absent_slot_selected'))
                    if decision.get('wall_ms') is not None:latencies.append(decision['wall_ms'])
            if kind=='control_frame':
                gate=row.get('plan_intent') or {};reasons[gate.get('reason','missing')]+=1
                counts['accepted_frames']+=bool(gate.get('accepted'))
                # Rank chosen by risk layer only counts when rollout won final arbiter.
                applied=bool(gate.get('accepted') and row.get('risk_reason')=='selected_model_rank_within_current_risk_budget'
                             and (row.get('arbitration') or {}).get('selected')=='rollout')
                counts['model_rank_final_frames']+=applied
    ordered=sorted(latencies)
    return {'schema':'jevt.plan_trace_diagnostic.v1','source':str(path),'counts':dict(counts),
            'gate_reasons':dict(reasons),'model_wall_ms':{'n':len(ordered),
                'p50':statistics.median(ordered) if ordered else None,
                'p95':ordered[min(len(ordered)-1,int(.95*len(ordered)))] if ordered else None},
            'runtime':runtime,
            'limits':'Observed runtime diagnostics, not model ablation or success probability. Final ranking usage is not counterfactual improvement.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trace',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=summarize(args.trace)
    if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
