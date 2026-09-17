# conversation-ru-v20

Authored synthetic Russian teaching material, not pretrained weights. Fixed262-ID tokenizer/model formatV1.
No private user chats or copied third-party story datasets are included. License follows this source project.

| Physical source split | Records | Use |
|---|---:|---|
| seed.jsonl |2966|Legacy main conversation/text source; extended by108records|
| conversation-starter.jsonl |281|NEW opt-in short conversational foundation|
| validation.jsonl |408|Control only;12new. Existing workspaces retain their own old controls|
| test/challenge files combined |308|Held-out;new challenge-v20 has8. Previous files unchanged|
| pretrain.jsonl |48|Frozen tiny original basic stage, unchanged|
| TOTAL |4011|Includes controls/tests;not4011training examples|

Main split:1718single-turn,814multi-turn,434texts. New108main records include16scenes represented as text,
question/answer and dialogue(48views),30new short coherent paragraphs and30three-exchange conversations.
Scene representations are related, not independent plots. Starter281includes paraphrases grouped by intent,
short replies and28short editing/grammar tasks. Do not count paraphrases as independent evidence of transfer.

The course uses3247original rows(main+starter). All-assistant-prefix preparation adds1320new training views
for this bundled set,4567total before replay. These are targets extracted from existing history, not new
source dialogues. Original supervised positions284199 become326472; context computation also increases.
Per-workspace control exclusions/data can change these counts. Actual worker reports preparation counters.

All current full source sequences<=453byte-token input positions, so default512fits without cropping.
Only training files enter gradients. Exact/normalized/full-input partition checks and manifest SHA256/count/
byte lengths are verified. They are NOT semantic independence tests. Related topics intentionally exist
across splits, and earlier challenge files are public. Public ConversationProbe is development evaluation,
not a new blind corpus or training material. Generated reports are NEVER automatically imported.

All historical held-out files from v19 and the48basictexts are byte-frozen. Deterministic builder:
python tools/build_conversation_dataset.py; validator:python tools/validate_dataset.py.
The authored v20 source records are in tools/corpus_v20_additions.json. Reproduction requires no network.

Teacher token accuracy, known-prompt recall and script invariants do not establish free conversation.
See reports/conversation-foundation-v20.json and conversation-full-v20.json for actual independent text outputs
and experimental limitations. These Python results do not certify C#/native/UI/CUDA behavior.
