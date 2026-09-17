# conversation-ru-v21 dataset card

Original authored synthetic/template data; MIT as the application. No personal chats, scraped text or
external language-model responses were imported. The source generator is tools/corpus_v21.py alongside
all previous version generators. Re-running build_conversation_dataset.py is deterministic.

| Section | Source rows | Purpose |
|---|---:|---|
| seed.jsonl |2990|Main TRAIN, +24 authored natural examples|
| conversation-starter.jsonl |281|Frozen optional short-answer TRAIN fromv20|
| conversation-context.jsonl |388|New explicitly template-labelled optional context TRAIN|
| validation.jsonl |408|Frozen control, NEVER gradients|
| existing test/challenge files |308|All19 historical heldout files frozen|
| context-challenge.jsonl |32|16new public counterfactual development pairs, NEVER gradients|
| pretrain.jsonl |48|Frozen separate minimal baseline TRAIN|
| TOTAL |4455|Not4455training examples or independent stories|

The opt-in source course union is3659rows before assistant-prefix expansion; the independent content-key
expansion yields5387views, including796views from context exercises. Replay repeats REFERENCES and is not
new unique data. Existing native ID/normalization guards remain authoritative and tested separately.

Training context templates vary names, other speakers, replacements, colors, moved objects, action status
and unknown facts. Paraphrase families and variant acknowledgements are related examples, not independent
facts. New public pair queries/introductions differ from training forms; names and domains intentionally
overlap. All generated earlier-assistant views of new test records are checked against training inputs.

Frozen hashes are in DATASET_MANIFEST.json and author evidence. All full examples fit512byte-tokens;
see reports/dataset-audit-v21.json for the measured maximum. Controls of existing workspaces are NOT replaced
when the application is upgraded. Importing a heldout split as training is rejected by the runtime.

Quality limitation: the6000-step independent medium-model experiment learns familiar context forms but
fails all16paired novel-form tasks. Dataset structural validity is NOT language quality or semantic split
independence. Public probes are diagnostic development evidence, not secret generalization certification.
