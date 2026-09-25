"""Local model routes; selecting a model never swaps out the C++ controller."""
from types import SimpleNamespace
from pathlib import Path
import json
import subprocess
import urllib.request


def read_json(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def laya_runtime(url):
    health=read_json(url.replace('/api/decision','/api/health'))
    if health.get('provider')!='CUDAExecutionProvider':
        raise RuntimeError('LAYA engine has not verified initialization of CUDAExecutionProvider')
    executable='/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi'
    name=subprocess.check_output([executable,'--id=0','--query-gpu=name','--format=csv,noheader'],
                                 text=True,timeout=5).strip()
    if '4090' not in name:
        raise RuntimeError('LAYA requires the RTX 4090 at device 0')
    return {'validated':True,'cuda_available':True,'device':'cuda:0','device_name':name,
            'provider':'CUDAExecutionProvider','model_kind':'laya','base_model_id':'laya-multilingual',
            'validation_scope':'CUDA session initialized; not a per-operator GPU residency audit'}


class ModelSelection:
    def __init__(self, selected, url, runtime):
        self.selected=selected
        self.options={key:{'id':key,'label':label,'available':False,'reason':'Not configured'}
                      for key,label in [('open_jev','Open-JEV'),('laya','LAYA')]}
        self.routes={selected:(url,runtime)}
        if selected in self.options:
            self.options[selected].update(available=True,reason='Ready')

    def snapshot(self):
        return {'selected':self.selected,'options':list(self.options.values()),
                'switch_semantics':'Change judgment backend; preserve lives, world, controller and same-run knowledge'}

    def select(self, key):
        if key not in self.routes:
            raise ValueError('Model unavailable: '+key)
        self.selected=key
        return self.routes[key]

    def add_laya(self, args, base, start_engine):
        path=getattr(args,'laya_model',None)
        if not path:
            return None
        if not Path(path).is_dir():
            self.options['laya']['reason']='LAYA bundle not found'
            return None
        alternate=SimpleNamespace(**vars(args))
        alternate.model=path;alternate.open_jev_url=None
        alternate.engine_port=args.engine_port+1
        alternate.model_url=f'http://127.0.0.1:{alternate.engine_port}/api/decision'
        process=None
        try:
            process=start_engine(alternate,base)
            runtime=laya_runtime(alternate.model_url)
            self.routes['laya']=(alternate.model_url.replace('/api/decision','/api/strategy'),runtime)
            self.options['laya'].update(available=True,reason='Ready · CUDA session initialized')
            return process
        except Exception as error:
            if process is not None:
                process.terminate()
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: process.kill();process.wait()
            self.options['laya']['reason']=str(error)
            return None
