#!/usr/bin/env python3
"""Known TRAINING-set recall after the independent experiment. Not held-out quality or C# execution.
The fixed index list is a diagnostic sample, not an exhaustive test. Each dialogue uses own generated history.
Requires the synthetic .pt produced by context_learning_experiment.py; never opens a user workspace.
"""
import argparse,hashlib,json,pathlib,torch
from context_learning_experiment import load,msgs,normal
from learning_reference import Model
from conversation_course_experiment import reply
INDICES=[0,5,15,30,60,125,160,185,205,240,300,330,360,387]
def run(state,output):
 torch.set_num_threads(2);m=Model(d=128,hid=384,n=4,heads=4,kv=2)
 m.load_state_dict(torch.load(state,weights_only=True,map_location='cpu'));rows=load('conversation-context.jsonl');out=[]
 for i in INDICES:
  r=rows[i];mm=msgs(r);history=[]
  for msg in mm[::2]:
   q=msg['content'];history.append((q,reply(m,q,history,limit=64)))
  teacher=[(mm[j]['content'],mm[j+1]['content'])for j in range(0,len(mm)-2,2)]
  final_teacher=reply(m,mm[-2]['content'],teacher,limit=64)
  out.append({'index':i,'id':r['id'],'expected_training_answer':mm[-1]['content'],'actual_history':history,
    'exact':normal(history[-1][1])==normal(mm[-1]['content']),'teacher_history_final':final_teacher})
 report={'scope':__doc__,'state_sha256':hashlib.sha256(pathlib.Path(state).read_bytes()).hexdigest(),
  'exact':sum(x['exact']for x in out),'total':len(out),'results':out}
 pathlib.Path(output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=run(a.weights,a.output);print(r['exact'],r['total'])
