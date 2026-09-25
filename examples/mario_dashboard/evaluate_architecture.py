"""Paired controller revision screen, explicitly not a model-quality benchmark.

Runs fresh emulator episodes with the same named stage/mode/timing conditions.
Frozen binary hashes identify what ran even when workspace source is changing.
"""
import argparse
import hashlib
import json
from pathlib import Path
from benchmark_controller import JsonlController, episode


def compare(before, after, mode):
    regressions=[]
    if before['flag_get'] and not after['flag_get']:regressions.append('lost_stage_clear')
    if mode=='hunter' and after['objectives']['confirmed_eliminations']<before['objectives']['confirmed_eliminations']:
        regressions.append('fewer_confirmed_eliminations')
    if mode=='score_attack' and after['objectives']['score']<before['objectives']['score']:
        regressions.append('lower_score')
    if after['max_x']<before['max_x']:regressions.append('less_progress')
    return {'max_x_delta':after['max_x']-before['max_x'],
            'kill_delta':after['objectives']['confirmed_eliminations']-before['objectives']['confirmed_eliminations'],
            'score_delta':after['objectives']['score']-before['objectives']['score'],
            'regression_signals':regressions}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--cases',default='SuperMarioBros-1-1-v0:speedrun,SuperMarioBros-1-2-v0:score_attack,SuperMarioBros-1-2-v0:hunter,SuperMarioBros-1-3-v0:speedrun')
    p.add_argument('--offsets',default='0,17');p.add_argument('--max-frames',type=int,default=3000)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    def checksum(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    binaries={'baseline':args.baseline,'candidate':args.candidate}
    hashes={key:checksum(path) for key,path in binaries.items()}
    results=[]
    for case in args.cases.split(','):
        environment,mode=case.rsplit(':',1)
        for offset in map(int,args.offsets.split(',')):
            pair={'environment':environment,'mode':mode,'offset':offset}
            # Counterbalance order; these are timing conditions, not random maps.
            order=list(binaries) if len(results)%2==0 else list(reversed(binaries))
            for revision in order:
                if checksum(binaries[revision])!=hashes[revision]:raise RuntimeError('Binary changed during evaluation')
                controller=JsonlController(binaries[revision],mode,'none',None)
                run=episode(0,offset,args.max_frames,False,controller,mode,0,environment)
                pair[revision]={k:v for k,v in run.items() if k not in {'trace','terminal_evidence','terminal_next_info'}}
            pair['comparison']=compare(pair['baseline'],pair['candidate'],mode)
            results.append(pair)
            print(json.dumps({k:v for k,v in pair.items() if k not in binaries}),flush=True)
    report={'schema':'jevt.paired_revision.v1','binary_sha256':hashes,'results':results,
            'provenance':'Fresh independent single-stage emulator episodes; no model inference, imported experience, save states or action replay.',
            'limits':'Descriptive paired screen, not statistical significance. Delays do not randomize maps and can produce identical runs. No full-game completion claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,default=lambda x:x.item()),encoding='utf-8')


if __name__=='__main__':main()
