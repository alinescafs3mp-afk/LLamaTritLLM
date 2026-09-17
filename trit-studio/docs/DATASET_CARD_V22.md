# conversation-ru-v22

Authored synthetic Russian teaching material. No private chats, copied external story corpus or pretrained weights.
Existing tokenizer262 IDs and binaryV1 unchanged. MIT source license. Deterministic builder uses tools/corpus_v22.py.

| Physical source | Records | Role |
|---|---:|---|
| seed.jsonl |2990|existing main train, BYTE-UNCHANGED|
| conversation-starter.jsonl |281|existing short train, BYTE-UNCHANGED|
| conversation-context.jsonl |388|existing context train, BYTE-UNCHANGED|
| conversation-language.jsonl |160|NEW20intent families x8formulations|
| conversation-transfer.jsonl |1278|NEW crossed fact/question training templates|
| validation.jsonl |408|unchanged control|
| all older heldout files |340|unchanged|
| transfer-challenge.jsonl |62|NEW public DEVELOPMENT evaluation, NOT train|
| pretrain.jsonl |48|unchanged original tiny basic stage|
| TOTAL |5955|not5955training dialogues|

The course has5097original TRAIN rows,7851views after all-assistant-target expansion before explicit sampling.
Priority language pool441original targets; priority fact pool1666original final targets. Expanded intermediate replies
remain in general corpus only. No duplicate buffers/strings are counted as new authored data. Actual workspace counts can differ after exclusions against its own unchanged controls or imported data.

New160 language rows use one stable short response per intention with8related phrasings. Their40development variants
use two additional authored forms per intent. Fact exercises cross names/intros/questions/corrections; reserved template
combinations give22additional cases. Familiar words and semantic tasks intentionally overlap; this is NOT proof of
semantic independence or a blind benchmark. Old public questions stay frozen and are still evaluated separately.

Known exact/full-history control and probe inputs, including derived assistant prefixes, were checked separately.
All whole sequences fit512; current corpus maximum453UTF8-byte input tokens. Stored controls of existing workspaces are
not replaced. The frozen48basic texts are NOT a broad Russian pretrain. Excluding a new test input cannot unlearn old data.

Generated first acknowledgements vary; supervised final facts are not forcibly inserted into model responses. Dataset
training and runtime inference are separate. New files enter only explicit TransferPractice, except evaluation files
which are read only by the evaluator. All adjacent app/checks/CPUtrainer/CUDAtrainer payload copies must update together.
