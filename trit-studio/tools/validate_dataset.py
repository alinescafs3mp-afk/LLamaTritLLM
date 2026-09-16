#!/usr/bin/env python3
"""Corpus checks only. Does not measure model quality or execute the C# tokenizer."""
from __future__ import annotations
import collections
import hashlib
import json
import pathlib
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parents[1]

def canonical(row):
    if 'messages' in row:
        messages = row['messages']
        assert len(messages) >= 2 and len(messages) % 2 == 0
        for i, msg in enumerate(messages):
            assert msg['role'] == ('user' if i % 2 == 0 else 'assistant')
        return messages
    if 'text' in row:
        return row['text']
    return [row['prompt'], row['answer']]

def strings(row):
    if 'messages' in row:
        return [x['content'] for x in row['messages']]
    if 'text' in row:
        return [row['text']]
    return [row['prompt'], row['answer']]

def normalized(value):
    if isinstance(value, str):
        return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    if isinstance(value, list):
        return [normalized(x) for x in value]
    return {k: normalized(v) for k, v in sorted(value.items())}

def full_length(row):
    values = strings(row)
    if 'messages' in row:
        return sum(len(x.encode('utf-8')) for x in values) + 4 * (len(values) // 2)
    return sum(len(x.encode('utf-8')) for x in values) + (1 if 'text' in row else 4)

def validate(write_report=True):
    manifest = json.loads((ROOT / 'data/DATASET_MANIFEST.json').read_text())
    seen, seen_normalized, prompt_split, input_split, canonical_inputs = {}, {}, {}, {}, {}
    assert hashlib.sha256((ROOT/'data/test.jsonl').read_bytes()).hexdigest() == manifest['frozen_v2_test_sha256']
    assert hashlib.sha256((ROOT/'data/challenge.jsonl').read_bytes()).hexdigest() == manifest['frozen_v3_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v4.jsonl').read_bytes()).hexdigest() == manifest['frozen_v4_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v5.jsonl').read_bytes()).hexdigest() == manifest['frozen_v5_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v6.jsonl').read_bytes()).hexdigest() == manifest['frozen_v6_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v7.jsonl').read_bytes()).hexdigest() == manifest['frozen_v7_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/pretrain.jsonl').read_bytes()).hexdigest() == manifest['frozen_baseline_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v8.jsonl').read_bytes()).hexdigest() == manifest['frozen_v8_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v9.jsonl').read_bytes()).hexdigest() == manifest['frozen_v9_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v10.jsonl').read_bytes()).hexdigest() == manifest['frozen_v10_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v11.jsonl').read_bytes()).hexdigest() == manifest['frozen_v11_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v12.jsonl').read_bytes()).hexdigest() == manifest['frozen_v12_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v13.jsonl').read_bytes()).hexdigest() == manifest['frozen_v13_challenge_sha256']
    assert hashlib.sha256((ROOT/'data/challenge-v14.jsonl').read_bytes()).hexdigest() == 'a404cb0ba9757ada7ecc80799734445ba17d70c299513214316535875244aa40'
    assert hashlib.sha256((ROOT/'data/challenge-v15.jsonl').read_bytes()).hexdigest() == '86262b63d04b1cdf2c609592e091adb2be4238e67d553f877b611b1a167e7c11'
    reports, overlaps = {}, []
    for split, label in [('seed', 'train'), ('validation', 'validation'), ('test', 'test'), ('challenge', 'test'), ('challenge-v4', 'test'), ('challenge-v5', 'test'), ('challenge-v6', 'test'), ('challenge-v7', 'test'), ('challenge-v8', 'test'), ('challenge-v9', 'test'), ('challenge-v10', 'test'), ('challenge-v11', 'test'), ('challenge-v12', 'test'), ('challenge-v13', 'test'), ('challenge-v14', 'test'), ('challenge-v15', 'test'), ('challenge-v16', 'test'), ('pretrain', 'train')]:
        path = ROOT / 'data' / (split + '.jsonl')
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == manifest['files'][split]['sha256'], f'Hash mismatch: {path}'
        rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        assert len(rows) == manifest['counts'][split] == manifest['files'][split]['count']
        kinds = collections.Counter()
        families = collections.Counter()
        lengths = []
        for i, row in enumerate(rows):
            assert row['split'] == label and row['origin'] in ('authored-synthetic-v2','authored-synthetic-v3','authored-synthetic-v4','authored-synthetic-v5','authored-synthetic-v6','authored-synthetic-v7','authored-synthetic-v8','authored-synthetic-v9','authored-synthetic-v10','authored-synthetic-v11','authored-synthetic-v12','authored-synthetic-v13','authored-synthetic-v14','authored-synthetic-v15','authored-synthetic-v16')
            value = canonical(row)
            identity = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            assert row['id'] == ('ru-'+row['origin'].rsplit('-',1)[-1]+'-') + identity[:20]
            assert identity not in seen, f'Exact duplicate: {split}:{i} and {seen.get(identity)}'
            seen[identity] = (split, i)
            norm = json.dumps(normalized(value), ensure_ascii=False, sort_keys=True)
            assert norm not in seen_normalized, f'Normalized duplicate: {split}:{i} and {seen_normalized.get(norm)}'
            seen_normalized[norm] = (split, i)
            for text in strings(row):
                assert isinstance(text, str) and text.strip() == text and text
                assert '\ufffd' not in text and '\x00' not in text
                text.encode('utf-8', errors='strict')
                assert not any(unicodedata.category(ch) == 'Cc' and ch not in '\n\t' for ch in text)
            if 'text' not in row:
                inputs = row['messages'][:-1] if 'messages' in row else row['prompt']
                input_key = json.dumps(normalized(inputs), ensure_ascii=False, sort_keys=True)
                assert input_key not in input_split or input_split[input_key] == split, f'Cross-split input/context reused at {split}:{i}'
                input_split[input_key] = split
            # Match runtime input identity across supported JSON representations. A one-pair
            # messages array and {prompt,answer} must not hide the same held-out input.
            if 'text' in row:
                runtime_input = ['text', row['text'], []]
            elif 'messages' in row:
                m = row['messages']
                runtime_input = ['dialogue', m[-2]['content'], [[m[j]['content'], m[j+1]['content']] for j in range(0,len(m)-2,2)]]
            else:
                runtime_input = ['dialogue', row['prompt'], []]
            runtime_key = json.dumps(normalized(runtime_input), ensure_ascii=False, sort_keys=True)
            assert runtime_key not in canonical_inputs or canonical_inputs[runtime_key] == label, f'Runtime-equivalent input crossed training/control/test: {split}:{i}'
            canonical_inputs[runtime_key] = label
            length = full_length(row)
            assert length <= 512, f'Target/context truncation required at {split}:{i}: {length}'
            lengths.append(length)
            kind = 'multi-turn' if 'messages' in row else 'plain-text' if 'text' in row else 'single-turn'
            kinds[kind] += 1; families[row['family']] += 1
            if kind == 'single-turn':
                prompt = normalized(row['prompt'])
                if prompt in prompt_split and prompt_split[prompt] != split:
                    overlaps.append({'prompt': row['prompt'], 'splits': [prompt_split[prompt], split]})
                prompt_split[prompt] = split
        reports[split] = {'count': len(rows), 'kinds': dict(kinds), 'families': len(families), 'max_sequence_bytes_with_roles': max(lengths),
            'mean_sequence_bytes_with_roles': round(sum(lengths)/len(lengths),2), 'sha256': hashlib.sha256(raw).hexdigest()}
    # Do not hide leakage behind a claim that only exact entire conversations were checked.
    assert not overlaps, 'Cross-split repeated single-turn prompts: ' + repr(overlaps)
    report = {'status':'passed', 'scope':'Data structure, integrity, disjointness and sequence budgeting; NOT a model quality evaluation',
              'corpus_version': manifest['version'], 'splits':reports, 'cross_split_normalized_duplicates':0, 'cross_split_exact_single_turn_prompts':0,
              'runtime_equivalent_cross_partition_inputs':0, 'semantic_generalization':'Unmeasured. Related everyday topics intentionally occur across splits.',
              'frozen_v2_test_unchanged':True, 'frozen_v3_challenge_unchanged':True, 'frozen_v4_challenge_unchanged':True, 'frozen_v5_challenge_unchanged':True, 'frozen_v6_challenge_unchanged':True, 'frozen_v7_challenge_unchanged':True, 'frozen_v8_challenge_unchanged':True, 'frozen_v9_challenge_unchanged':True, 'frozen_v10_challenge_unchanged':True, 'frozen_v11_challenge_unchanged':True, 'frozen_v12_challenge_unchanged':True, 'frozen_v13_challenge_unchanged':True, 'frozen_v14_challenge_unchanged':True, 'frozen_v15_challenge_unchanged':True, 'baseline_v7_unchanged':True, 'cross_split_input_context_duplicates':0, 'total_examples':sum(x['count'] for x in reports.values())}
    if write_report:
        (ROOT/'reports').mkdir(exist_ok=True)
        (ROOT/'reports/dataset-audit-v16.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report

if __name__ == '__main__':
    try: print(json.dumps(validate(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print('DATASET AUDIT FAILED: '+str(exc),file=sys.stderr); raise
