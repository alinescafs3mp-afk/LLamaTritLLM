#!/usr/bin/env python3
"""Black-box acceptance smoke for the ACTUAL compiled trainer. Standard-library only.
Usage: python3 tools/worker_smoke.py dist/linux-x64-cpu/trainer/TritStudio.Trainer
No test is marked passed unless a trainer process ran and its files were inspected.
"""
import argparse, hashlib, json, pathlib, queue, subprocess, tempfile, threading, time, uuid

def example(text,answer=None):
    # .NET's default JSON encoder escapes non-ASCII, so use ASCII in identity-sensitive smoke inputs.
    ident=hashlib.sha256(json.dumps([text,answer],separators=(',',':'),ensure_ascii=True).encode()).hexdigest().upper()
    return dict(id=ident,text=text,answer=answer,source='smoke')
class Worker:
    def __init__(self, executable, workspace):
        self.q=queue.Queue();self.events=[];self.stderr=[]
        self.proc=subprocess.Popen([str(executable),'--workspace',str(workspace)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',bufsize=1)
        def read():
            for line in self.proc.stdout:
                try:self.q.put(json.loads(line))
                except ValueError:self.q.put(dict(kind='invalid',data=line))
            self.q.put(dict(kind='exit',data=self.proc.poll()))
        def err():
            for line in self.proc.stderr:self.stderr.append(line.rstrip())
        threading.Thread(target=read,daemon=True).start();threading.Thread(target=err,daemon=True).start()
    def send(self,kind,payload):
        ident=uuid.uuid4().hex
        self.proc.stdin.write(json.dumps(dict(kind=kind,id=ident,payload=payload),separators=(',',':'))+'\n');self.proc.stdin.flush();return ident
    def wait(self,predicate,timeout=180):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            try:event=self.q.get(timeout=max(.01,deadline-time.monotonic()))
            except queue.Empty:break
            self.events.append(event)
            if event['kind'] in ('error','invalid','exit'):raise AssertionError(f"Worker failed: {event}; stderr={self.stderr[-10:]}")
            if predicate(event):return event
        raise TimeoutError(f"Trainer event timed out. Last events={self.events[-10:]}; stderr={self.stderr[-10:]}")
    def close(self):
        if self.proc.poll() is None:
            self.send('shutdown',{})
            self.proc.stdin.close()
            try:self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait();raise
        if self.proc.returncode!=0:raise AssertionError(f'Worker exit {self.proc.returncode}: {self.stderr[-10:]}')

def run(executable):
    with tempfile.TemporaryDirectory(prefix='trit-worker-smoke-') as temp:
        path=pathlib.Path(temp);w=Worker(executable,path)
        try:
            w.wait(lambda e:e['kind']=='ready')
            cfg=dict(dimension=16,hiddenDimension=32,layers=1,heads=2,kvHeads=1,context=512,planes=2,groupSize=8,threshold=.5,seed=42)
            resources=dict(threads=1,memoryMiB=4096,batchSize=1,sequenceLength=384,preferCuda=False)
            training=dict(steps=2,publishEvery=2,learningRate=.0001,onlineLearningRate=.00005,maxValidationRegression=.5)
            job=w.send('create',dict(config=cfg,resources=resources,training=training,datasetPaths=[]))
            w.wait(lambda e:e['kind']=='completed' and e.get('id')==job)
            active=json.loads((path/'active.json').read_text())['directory']
            initial_info=json.loads((path/'revisions'/active/'revision.json').read_text());assert initial_info['step']==2
            w.send('mode',dict(enabled=True,learningRate=.00005))
            ex=example('hello','hi');w.send('online',dict(example=ex))
            published=w.wait(lambda e:e['kind']=='published' and e['data']['info']['reason']=='Онлайн-обучение')
            info=published['data']['info'];assert info['changedWeights']>0 and info['step']==6
            w.wait(lambda e:e['kind']=='status' and e['data']['queue']==0)
            online_name=published['data']['revisionDirectory']
            job=w.send('rollback',{});w.wait(lambda e:e['kind']=='completed' and e.get('id')==job)
            assert json.loads((path/'active.json').read_text())['directory']==active
            ledger=json.loads((path/'replay.json').read_text());assert ledger['entries'][0]['state']=='rolledback'
            job=w.send('online',dict(example=ex));w.wait(lambda e:e['kind']=='completed' and e.get('id')==job)
            assert json.loads((path/'active.json').read_text())['directory']==active
            w.close()
            w=Worker(executable,path);w.wait(lambda e:e['kind']=='ready' and e['data']['hasModel'])
            assert json.loads((path/'active.json').read_text())['directory']==active
            print(json.dumps(dict(status='passed',scope='actual C# worker',online_changed_weights=info['changedWeights'],offline_revision=active,online_revision=online_name,
                checks=['handshake without input deadlock','one-command create/train/publish','real online master-weight changes','rollback of weights and optimizer state','no duplicate application after rollback','restart loads active checkpoint']),indent=2))
        finally:w.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('trainer',type=pathlib.Path);args=ap.parse_args()
    executable=args.trainer.resolve()
    if not executable.is_file():raise SystemExit('Compiled trainer not found; smoke was NOT run.')
    run(executable)
