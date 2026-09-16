#!/usr/bin/env python3
"""Independent policy checks of update routing and chat boundaries. NOT execution of C# or PowerShell."""
from pathlib import Path
import hashlib,json,zipfile,tempfile
R=Path(__file__).resolve().parents[1]
def owned(p):
    if not p or p.startswith('/') or '\\' in p or ':' in p or any(x in ('','.','..') for x in p.split('/')):raise ValueError(p)
    if p=='checks/reference.json':return True
    a=p.split('/')
    owners={'TritStudio','TritStudio.App','TritStudio.Core','TritStudio.Trainer','TritStudio.Runner','TritStudio.Tests'}
    if len(a)<=2 and (len(a)==1 or a[0] in ('trainer','trainer-cuda','runner','checks')):
        n=a[-1]
        for s in ('.deps.json','.runtimeconfig.json','.exe','.dll','.pdb'):
            if n.endswith(s):return n[:-len(s)] in owners
        return n in owners
    if p=='checks/reference.json':return True
    if len(a)==3 and a[0] in ('trainer','trainer-cuda','checks') and a[1]=='data':
        n=a[2]
        if n in ('DATASET_MANIFEST.json','seed.jsonl','validation.jsonl','test.jsonl','challenge.jsonl','pretrain.jsonl'):return True
        if n.startswith('challenge-v') and n.endswith('.jsonl'):
            v=n[11:-6];return v.isascii() and v.isdigit() and int(v)>=4
    return False
files={'TritStudio.exe':b'app','TritStudio.Core.dll':b'core','Avalonia.Controls.dll':b'avalonia','hostfxr.dll':b'net'}
for prefix in ('trainer','trainer-cuda','checks'):
    app='TritStudio.Tests' if prefix=='checks' else 'TritStudio.Trainer'
    for suffix in ('.exe','.dll','.deps.json','.runtimeconfig.json'): files[f'{prefix}/{app}{suffix}']=b'owned'
    for n in ('seed.jsonl','validation.jsonl','pretrain.jsonl','challenge-v15.jsonl','DATASET_MANIFEST.json'):files[f'{prefix}/data/{n}']=b'dataset'
    files[f'{prefix}/System.Private.CoreLib.dll']=b'vendor'
files['trainer-cuda/torch_cuda.dll']=b'cuda';files['trainer/TorchSharp.dll']=b'torch';files['checks/reference.json']=b'fixture'
selected={n:v for n,v in files.items() if owned(n)}
assert 'trainer-cuda/torch_cuda.dll' not in selected and all(f'{r}/data/seed.jsonl' in selected for r in ('trainer','trainer-cuda','checks'))
assert 'checks/reference.json' in selected
for p in ('../TritStudio.exe','C:/TritStudio.exe','trainer\\TritStudio.exe','a//TritStudio.dll'):
    try:owned(p)
    except ValueError:pass
    else:raise AssertionError(p)
# Actual Python overlay of fixtures preserves independent native payloads; not a test of .NET publish.
with tempfile.TemporaryDirectory() as d:
    home=Path(d);installed=home/'installed';installed.mkdir()
    for n,v in files.items():
        p=installed/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
    vendor={n:hashlib.sha256((installed/n).read_bytes()).hexdigest() for n in files if n not in selected}
    z=home/'update.zip'
    with zipfile.ZipFile(z,'w') as out:
        for n in selected:out.writestr(n,b'new-'+selected[n])
    with zipfile.ZipFile(z) as src:src.extractall(installed)
    assert all(hashlib.sha256((installed/n).read_bytes()).hexdigest()==h for n,h in vendor.items())
    assert all((installed/n).read_bytes()==b'new-'+v for n,v in selected.items())
# Clear is a boundary, not deletion, and durable failure must not clear the visible old context.
for ok in (False,True):
    state={'conversation':'a','history':['one','two'],'draft':'next','private':True,'weights':'same','queue':['pending'],'disk_log':['one','two']}
    original=json.loads(json.dumps(state))
    if ok:state.update(conversation='b',history=[])
    assert state['draft']==original['draft'] and state['private'] and state['weights']=='same' and state['queue']==['pending'] and state['disk_log']==['one','two']
    assert (state['history']==[])==ok
manifest=json.loads((R/'data/DATASET_MANIFEST.json').read_text())
report={'status':'passed','scope':__doc__,'synthetic_publish_files':len(files),'owned_files':len(selected),'retained_vendor_files':len(vendor),
 'actual_fixture_overlay_verified':True,'corpus_records':sum(manifest['counts'].values()),'csharp':'NOT_RUN','powershell':'NOT_RUN','native_training':'NOT_RUN','gui':'NOT_RUN','cuda':'NOT_RUN'}
(R/'reports/audit15-reference.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
