#!/usr/bin/env python3
"""Deterministic v18 corpus builder. The complete v2 test set is byte-for-byte frozen.
No external calls, no private conversations. Edit corpus_v18_additions.json for reviewed additions.
"""
from pathlib import Path
import hashlib, json
import corpus_v2 as old
import corpus_v3_additions as new
import corpus_v4_additions as current
import corpus_v5_additions as latest
import corpus_v6_additions as audit6
import corpus_v7_additions as audit7
import corpus_v8_additions as audit8
import corpus_v9_additions as audit9
import corpus_v10_additions as audit10
import corpus_v11_additions as audit11
import corpus_v12_additions as audit12
import corpus_v13_additions as audit13
import corpus_v14_additions as audit14
ROOT = Path(__file__).resolve().parents[1]
FROZEN_V2_TEST_SHA256 = "cc828879d8996742756520d20a5971106e4ecb099e9406690459e414eea508fb"

def additions(text, split, multi=False, version=3):
    family = None
    for line in text.strip().splitlines():
        if line.startswith('['): family=line.strip('[]');continue
        assert family and line.strip(), line
        parts=line.split('|')
        if version == 14: parts=[part.replace(r'\n', '\n') for part in parts]
        if multi:
            assert len(parts) in (4,6,8), line
            row={'messages':[{'role':'user' if i%2==0 else 'assistant','content':s} for i,s in enumerate(parts)]}
        else:
            assert len(parts)==2, line
            row={'prompt':parts[0],'answer':parts[1]}
        row.update(family=family, split=split, origin=f'authored-synthetic-v{version}')
        row['id']=f'ru-v{version}-'+old.identity(row)[:20]
        yield row

def legacy():
    train=old.parse(old.TRAIN,'train')+list(old.multi_rows(old.MULTI,'train'))
    train += [{'text':x,'family':f'pretrain_{i:03}','split':'train','origin':'authored-synthetic-v2'} for i,x in enumerate(old.TEXTS.strip().splitlines())]
    val=old.parse(old.VALIDATION,'validation')+list(old.multi_rows(old.VAL_MULTI,'validation'))
    test=old.parse(old.TEST,'test')+list(old.multi_rows(old.TEST_MULTI,'test'))
    for rows in [train,val,test]:
        for r in rows:r['id']='ru-v2-'+old.identity(r)[:20]
    return train,val,test

def run():
    train,validation,test=legacy()
    train+=list(additions(new.PAIRS,'train'))+list(additions(new.MULTI,'train',True))
    for i,line in enumerate(new.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v3_plain_{i:03}','split':'train','origin':'authored-synthetic-v3'}
        row['id']='ru-v3-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(new.VALIDATION,'validation'))+list(additions(new.VAL_MULTI,'validation',True))
    challenge=list(additions(new.CHALLENGE,'test'))+list(additions(new.CHALLENGE_MULTI,'test',True))
    train+=list(additions(current.PAIRS,'train',version=4))+list(additions(current.MULTI,'train',True,version=4))
    for i,line in enumerate(current.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v4_plain_{i:03}','split':'train','origin':'authored-synthetic-v4'}
        row['id']='ru-v4-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(current.VALIDATION,'validation',version=4))+list(additions(current.VAL_MULTI,'validation',True,version=4))
    challenge4=list(additions(current.CHALLENGE,'test',version=4))+list(additions(current.CHALLENGE_MULTI,'test',True,version=4))
    train+=list(additions(latest.PAIRS,'train',version=5))+list(additions(latest.MULTI,'train',True,version=5))
    for i,line in enumerate(latest.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v5_plain_{i:03}','split':'train','origin':'authored-synthetic-v5'}
        row['id']='ru-v5-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(latest.VALIDATION,'validation',version=5))+list(additions(latest.VAL_MULTI,'validation',True,version=5))
    challenge5=list(additions(latest.CHALLENGE,'test',version=5))+list(additions(latest.CHALLENGE_MULTI,'test',True,version=5))
    train+=list(additions(audit6.PAIRS,'train',version=6))+list(additions(audit6.MULTI,'train',True,version=6))
    for i,line in enumerate(audit6.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v6_plain_{i:03}','split':'train','origin':'authored-synthetic-v6'}
        row['id']='ru-v6-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit6.VALIDATION,'validation',version=6))+list(additions(audit6.VAL_MULTI,'validation',True,version=6))
    challenge6=list(additions(audit6.CHALLENGE,'test',version=6))+list(additions(audit6.CHALLENGE_MULTI,'test',True,version=6))
    train+=list(additions(audit7.PAIRS,'train',version=7))+list(additions(audit7.MULTI,'train',True,version=7))
    for i,line in enumerate(audit7.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v7_plain_{i:03}','split':'train','origin':'authored-synthetic-v7'}
        row['id']='ru-v7-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit7.VALIDATION,'validation',version=7))+list(additions(audit7.VAL_MULTI,'validation',True,version=7))
    challenge7=list(additions(audit7.CHALLENGE,'test',version=7))+list(additions(audit7.CHALLENGE_MULTI,'test',True,version=7))
    train+=list(additions(audit8.PAIRS,'train',version=8))+list(additions(audit8.MULTI,'train',True,version=8))
    for i,line in enumerate(audit8.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v8_plain_{i:03}','split':'train','origin':'authored-synthetic-v8'}
        row['id']='ru-v8-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit8.VALIDATION,'validation',version=8))+list(additions(audit8.VAL_MULTI,'validation',True,version=8))
    challenge8=list(additions(audit8.CHALLENGE,'test',version=8))+list(additions(audit8.CHALLENGE_MULTI,'test',True,version=8))
    train+=list(additions(audit9.PAIRS,'train',version=9))+list(additions(audit9.MULTI,'train',True,version=9))
    for i,line in enumerate(audit9.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v9_plain_{i:03}','split':'train','origin':'authored-synthetic-v9'}
        row['id']='ru-v9-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit9.VALIDATION,'validation',version=9))+list(additions(audit9.VAL_MULTI,'validation',True,version=9))
    challenge9=list(additions(audit9.CHALLENGE,'test',version=9))+list(additions(audit9.CHALLENGE_MULTI,'test',True,version=9))
    train+=list(additions(audit10.PAIRS,'train',version=10))+list(additions(audit10.MULTI,'train',True,version=10))
    for i,line in enumerate(audit10.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v10_plain_{i:03}','split':'train','origin':'authored-synthetic-v10'}
        row['id']='ru-v10-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit10.VALIDATION,'validation',version=10))+list(additions(audit10.VAL_MULTI,'validation',True,version=10))
    challenge10=list(additions(audit10.CHALLENGE,'test',version=10))+list(additions(audit10.CHALLENGE_MULTI,'test',True,version=10))
    train+=list(additions(audit11.PAIRS,'train',version=11))+list(additions(audit11.MULTI,'train',True,version=11))
    for i,line in enumerate(audit11.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v11_plain_{i:03}','split':'train','origin':'authored-synthetic-v11'}
        row['id']='ru-v11-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit11.VALIDATION,'validation',version=11))+list(additions(audit11.VAL_MULTI,'validation',True,version=11))
    challenge11=list(additions(audit11.CHALLENGE,'test',version=11))+list(additions(audit11.CHALLENGE_MULTI,'test',True,version=11))
    train+=list(additions(audit12.PAIRS,'train',version=12))+list(additions(audit12.MULTI,'train',True,version=12))
    for i,line in enumerate(audit12.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v12_plain_{i:03}','split':'train','origin':'authored-synthetic-v12'}
        row['id']='ru-v12-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit12.VALIDATION,'validation',version=12))+list(additions(audit12.VAL_MULTI,'validation',True,version=12))
    challenge12=list(additions(audit12.CHALLENGE,'test',version=12))+list(additions(audit12.CHALLENGE_MULTI,'test',True,version=12))
    train+=list(additions(audit13.PAIRS,'train',version=13))+list(additions(audit13.MULTI,'train',True,version=13))
    for i,line in enumerate(audit13.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v13_plain_{i:03}','split':'train','origin':'authored-synthetic-v13'}
        row['id']='ru-v13-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit13.VALIDATION,'validation',version=13))+list(additions(audit13.VAL_MULTI,'validation',True,version=13))
    challenge13=list(additions(audit13.CHALLENGE,'test',version=13))+list(additions(audit13.CHALLENGE_MULTI,'test',True,version=13))
    train+=list(additions(audit14.PAIRS,'train',version=14))+list(additions(audit14.MULTI,'train',True,version=14))
    for i,line in enumerate(audit14.TEXTS.strip().splitlines()):
        row={'text':line,'family':f'v14_plain_{i:03}','split':'train','origin':'authored-synthetic-v14'}
        row['id']='ru-v14-'+old.identity(row)[:20];train.append(row)
    validation+=list(additions(audit14.VALIDATION,'validation',version=14))+list(additions(audit14.VAL_MULTI,'validation',True,version=14))
    challenge14=list(additions(audit14.CHALLENGE,'test',version=14))+list(additions(audit14.CHALLENGE_MULTI,'test',True,version=14))
    new15=json.loads((ROOT/'tools/corpus_v15_additions.json').read_text())
    train+=new15['train']; validation+=new15['validation']; challenge15=new15['challenge']
    new16=json.loads((ROOT/'tools/corpus_v16_additions.json').read_text())
    train+=new16['train']; validation+=new16['validation']; challenge16=new16['challenge']
    new17=json.loads((ROOT/'tools/corpus_v17_additions.json').read_text())
    train+=new17['train']; validation+=new17['validation']; challenge17=new17['challenge']
    new18=json.loads((ROOT/'tools/corpus_v18_additions.json').read_text())
    train+=new18['train']; validation+=new18['validation']; challenge18=new18['challenge']
    pretrain=[]
    for i,line in enumerate(audit7.BASELINE.strip().splitlines()):
        row={'text':line,'family':f'v7_baseline_{i:03}','split':'train','origin':'authored-synthetic-v7'}
        row['id']='ru-v7-'+old.identity(row)[:20];pretrain.append(row)
    outputs={name:''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows).encode('utf-8') for name,rows in [('seed',train),('validation',validation),('test',test),('challenge',challenge),('challenge-v4',challenge4),('challenge-v5',challenge5),('challenge-v6',challenge6),('challenge-v7',challenge7),('challenge-v8',challenge8),('challenge-v9',challenge9),('challenge-v10',challenge10),('challenge-v11',challenge11),('challenge-v12',challenge12),('challenge-v13',challenge13),('challenge-v14',challenge14),('challenge-v15',challenge15),('challenge-v16',challenge16),('challenge-v17',challenge17),('challenge-v18',challenge18),('pretrain',pretrain)]}
    assert hashlib.sha256(outputs['test']).hexdigest()==FROZEN_V2_TEST_SHA256, 'Frozen v2 test changed'
    frozen_v3_challenge = '2aa3d83f5cf51b85a0ee18e9e35e65e065ae7672dea573844abd676756fab550'
    assert hashlib.sha256(outputs['challenge']).hexdigest()==frozen_v3_challenge, 'Frozen v3 challenge changed'
    frozen_v4_challenge = '28cfbd9742401023153750dc7e0145d222a48189d8d9330978332ff134f0fadf'
    assert hashlib.sha256(outputs['challenge-v4']).hexdigest()==frozen_v4_challenge, 'Frozen v4 challenge changed'
    frozen_v5_challenge = '3493915e807d59feb388d7dff814cd6910db20d5f797c324dca5319bacf8223e'
    assert hashlib.sha256(outputs['challenge-v5']).hexdigest()==frozen_v5_challenge, 'Frozen v5 challenge changed'
    frozen_v6_challenge = '967b7c5e3b2bda83165e1f8b278ef5f70b74ba8b9ad4c588ea487f4882ee791e'
    assert hashlib.sha256(outputs['challenge-v6']).hexdigest()==frozen_v6_challenge, 'Frozen v6 challenge changed'
    frozen_v7_challenge='ee7072895a15764312986060d9b430b9fc56227966f5acf020bbaaa9f3bebfd8'
    frozen_baseline='a1efc4e62aba0afbe0d2f062e76e8f68c8f54454c68fe985a88ece58a1d14973'
    assert hashlib.sha256(outputs['challenge-v7']).hexdigest()==frozen_v7_challenge, 'Frozen v7 challenge changed'
    assert hashlib.sha256(outputs['pretrain']).hexdigest()==frozen_baseline, 'Frozen baseline changed'
    assert hashlib.sha256(outputs['challenge-v12']).hexdigest()=='462930a49fdffeab019d231da1c960d7906343476bdf3b96fc3865787baa0d32', 'Frozen v12 challenge changed'
    frozen_v8_challenge='5c07ee49983aeafa71a83196dcb5762ebbebb7828d00c76885420c0ab9370743'
    assert hashlib.sha256(outputs['challenge-v8']).hexdigest()==frozen_v8_challenge, 'Frozen v8 challenge changed'
    frozen_v9_challenge='fccb39e40af542c4d1b863f16978519868ac88cff3356d1320ccf2cb47e26e57'
    assert hashlib.sha256(outputs['challenge-v9']).hexdigest()==frozen_v9_challenge, 'Frozen v9 challenge changed'
    frozen_v10_challenge='8ac9e4ef854e1f9e953e07697fda8c7e2acbf7d441a8bd9b4712f1d95bcf6020'
    assert hashlib.sha256(outputs['challenge-v10']).hexdigest()==frozen_v10_challenge, 'Frozen v10 challenge changed'
    frozen_v11_challenge='9fdd58708166d1259ee56462e00de255c737477611a51e238dd0d2a148a37885'
    assert hashlib.sha256(outputs['challenge-v11']).hexdigest()==frozen_v11_challenge, 'Frozen v11 challenge changed'
    assert hashlib.sha256(outputs['challenge-v13']).hexdigest()=='888d908276483030ca818c74d0acaca65a7da9ebedad5cc1762509d260495ef3', 'Frozen v13 challenge changed'
    assert hashlib.sha256(outputs['challenge-v14']).hexdigest()=='a404cb0ba9757ada7ecc80799734445ba17d70c299513214316535875244aa40', 'Frozen v14 challenge changed'
    assert hashlib.sha256(outputs['challenge-v15']).hexdigest()=='86262b63d04b1cdf2c609592e091adb2be4238e67d553f877b611b1a167e7c11', 'Frozen v15 challenge changed'
    assert hashlib.sha256(outputs['challenge-v16']).hexdigest()=='125495efa5ee8d3dabfa7ddde8898486d807e53388d1da7f6ae5bd0b651c93d0', 'Frozen v16 challenge changed'
    assert hashlib.sha256(outputs['challenge-v17']).hexdigest()=='efbb9ea61308bca0cf3e9f9ddeb0346e5a7f802527638b053699244b9e20968e', 'Frozen v17 challenge changed'
    files={}
    for name,raw in outputs.items():
        (ROOT/'data'/f'{name}.jsonl').write_bytes(raw)
        files[name]={'count':len(raw.decode().splitlines()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    manifest={'version':'conversation-ru-v18','language':'ru','origin':'original authored synthetic text; no private chats or scraping',
      'license':'MIT (same as application)','counts':{k:v['count'] for k,v in files.items()},'files':files,
      'blind_test':'test.jsonl, challenge.jsonl, challenge-v4.jsonl, challenge-v5.jsonl challenge-v6.jsonl challenge-v7.jsonl challenge-v8.jsonl, challenge-v9.jsonl challenge-v10.jsonl challenge-v11.jsonl challenge-v12.jsonl challenge-v13.jsonl challenge-v14.jsonl challenge-v15.jsonl challenge-v16.jsonl challenge-v17.jsonl and challenge-v18.jsonl NEVER enter training; prior held-out generations are byte-for-byte frozen',
      'frozen_v2_test_sha256':FROZEN_V2_TEST_SHA256, 'frozen_v3_challenge_sha256':frozen_v3_challenge,'frozen_v4_challenge_sha256':frozen_v4_challenge,'frozen_v5_challenge_sha256':frozen_v5_challenge,'frozen_v6_challenge_sha256':frozen_v6_challenge,'frozen_v7_challenge_sha256':frozen_v7_challenge,'frozen_v8_challenge_sha256':frozen_v8_challenge,'frozen_v9_challenge_sha256':frozen_v9_challenge,'frozen_v10_challenge_sha256':frozen_v10_challenge,'frozen_v11_challenge_sha256':frozen_v11_challenge,'frozen_v12_challenge_sha256':hashlib.sha256(outputs['challenge-v12']).hexdigest(),'frozen_v13_challenge_sha256':'888d908276483030ca818c74d0acaca65a7da9ebedad5cc1762509d260495ef3','frozen_v14_challenge_sha256':'a404cb0ba9757ada7ecc80799734445ba17d70c299513214316535875244aa40','frozen_v15_challenge_sha256':'86262b63d04b1cdf2c609592e091adb2be4238e67d553f877b611b1a167e7c11','frozen_v16_challenge_sha256':'125495efa5ee8d3dabfa7ddde8898486d807e53388d1da7f6ae5bd0b651c93d0','frozen_v17_challenge_sha256':'efbb9ea61308bca0cf3e9f9ddeb0346e5a7f802527638b053699244b9e20968e','frozen_baseline_sha256':frozen_baseline,'baseline':'pretrain.jsonl is a separate text-only training stage; never injected into random initialization or claimed as pretrained weights','claim':'starter corpus, not pretrained weights or a conversational quality guarantee',
      'generation':'python3 tools/build_conversation_dataset.py','evolution':'Append reviewed train families, retain old held-out tests, add new held-out families separately.'}
    (ROOT/'data/DATASET_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest['counts']))
if __name__=='__main__':run()
