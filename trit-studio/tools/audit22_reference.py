#!/usr/bin/env python3
"""Independent Python algorithm/data checks, NOT C# native or UI execution."""
from __future__ import annotations
import collections,hashlib,json,pathlib,random
from context_learning_experiment import expand
from conversation_full_experiment import key,msgs
from learning_corpus_experiment import encode_row
ROOT=pathlib.Path(__file__).resolve().parents[1]
PATTERNS=((1,1,0,1,2,1,0,2),(1,2,0,2,1,2,0,1),(1,0,2,1,0,2,0,1))
def load(n):return [json.loads(l)for l in (ROOT/'data'/n).read_text().splitlines()if l.strip()]
def check():
 files=['seed.jsonl','conversation-starter.jsonl','conversation-context.jsonl','conversation-language.jsonl','conversation-transfer.jsonl']
 controls=load('validation.jsonl');guard={key(r)for r in controls}
 rows=sum((load(n)for n in files),[]);views=expand(rows,guard);oldctx=load('conversation-context.jsonl');oldviews=expand(oldctx,guard)
 assert len(oldviews)==796 and len(oldctx)==388
 old_targets=collections.Counter(msgs(x)[-1]['content']for x in oldviews)
 inputs={key(r)for r in views};probes=load('transfer-challenge.jsonl')+load('context-challenge.jsonl')
 probeviews=expand(probes,set())
 assert not inputs.intersection(key(r)for r in probeviews)
 assert not inputs.intersection(guard)
 assert len(load('conversation-language.jsonl'))==160 and len(load('conversation-transfer.jsonl'))==1278
 assert len(load('transfer-challenge.jsonl'))==62
 for row in load('conversation-transfer.jsonl')+load('transfer-challenge.jsonl'):
  text=msgs(row)[-2]['content'].lower();assert 'мой имя' not in text and 'имя тебе известен' not in text
 qchecks=0
 for total in [1,2,3,7,10,100,101,100000]:
  for done in sorted({0,total//5,min(total-1,3*total//5),total-1}):
   ph=0 if done*5<total else 1 if done*5<3*total else 2
   for b in range(1,65):
    slots=[PATTERNS[ph][(done*b+i)%8]for i in range(b)];assert len(slots)==b
    counts=collections.Counter(slots);assert sum(counts.values())==b
    if b%8==0:
     unit=collections.Counter(PATTERNS[ph]);assert all(counts[k]==unit[k]*(b//8)for k in unit)
    qchecks+=1
 # One-row training still visits each component; no permanently starved component.
 assert collections.Counter(PATTERNS[0][i%8]for i in range(8))=={0:2,1:4,2:2}
 data={key(x,True):i for i,x in enumerate(views)};encoded=[encode_row(r)for r in views]
 language=[data[key(r,True)]for r in load('conversation-starter.jsonl')+load('conversation-language.jsonl')]
 facts=oldctx+load('conversation-transfer.jsonl');groups=collections.defaultdict(list)
 for row in facts:groups[msgs(row)[-2]['content']].append(data[key(row,True)])
 gs=list(groups.values());draws=0
 for b in [1,2,7,8,16,31,64]:
  a=random.Random(117);z=random.Random(117)
  for done in range(100):
   ph=0 if done<20 else 1 if done<60 else 2
   def select(rng):
    selected=[]
    for slot in range(b):
     k=PATTERNS[ph][(done*b+slot)%8]
     if k==0:i=rng.randrange(len(views))
     elif k==1:i=language[rng.randrange(len(language))]
     else:
      g=gs[rng.randrange(len(gs))];i=g[rng.randrange(len(g))]
     selected.append(i)
    return selected
   first=select(a);second=select(z);assert first==second and a.getstate()==z.getstate()
   targets=sum(sum(y>=0 for y in encoded[i][1])for i in first)
   independent=sum(1 for i in first for y in encoded[i][1]if y!=-100)
   assert targets==independent;draws+=1
 # All references fit complete source target/history; new exact targets fit the112-byte report limit.
 for row in rows+probes:assert len(encode_row(row)[0])<=512
 assert max(len(msgs(r)[-1]['content'].encode())for r in load('transfer-challenge.jsonl'))<=112
 report={'scope':'Independent Python data/policy/sampling specification, NOT C# RNG, TPL, native optimizer, UI or CUDA execution','status':'passed',
  'quota_cases':qchecks,'sampling_reproducibility_and_target_count_cases':draws,'source_course_rows':len(rows),'expanded_views':len(views),
  'old_context_original_final_targets':len(oldctx),'old_context_expanded_priority_views':len(oldviews),'old_intermediate_priority_targets':len(oldviews)-len(oldctx),
  'old_intermediate_fraction':(len(oldviews)-len(oldctx))/len(oldviews),'old_most_frequent_targets':old_targets.most_common(8),
  'new_language_priority_rows':len(language),'new_original_final_fact_priority_rows':len(facts),'final_question_groups':len(groups),
  'public_new_cases':len(load('transfer-challenge.jsonl')),'old_public_variants':len(load('context-challenge.jsonl')),
  'full_and_derived_probe_input_overlap':0,'full_control_input_overlap':0,'max_new_expected_bytes':max(len(msgs(r)[-1]['content'].encode())for r in load('transfer-challenge.jsonl')),
  'data_hashes':{n:hashlib.sha256((ROOT/'data'/n).read_bytes()).hexdigest()for n in files+['validation.jsonl','transfer-challenge.jsonl','context-challenge.jsonl']}}
 (ROOT/'reports/audit22-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2));return report
if __name__=='__main__':check()
