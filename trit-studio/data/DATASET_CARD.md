# conversation-ru-v15 dataset card

Original authored synthetic Russian starter material, not scraped text, private chat, pretrained weights or proof of fluency.
`python tools/build_conversation_dataset.py` reproducibly regenerates data using retained versioned source modules and v15 additions.

| Part | Records | Use |
|---|---:|---|
| seed.jsonl | 2634 | Main training material |
| validation.jsonl | 352 | Repeated candidate guard, NOT blind evaluation |
| test.jsonl and challenge*.jsonl | 256 | Held-out material, excluded from ordinary training |
| pretrain.jsonl | 48 | Separate frozen basic stage |
| Total | 3290 | Only training material enters gradients |

This pass adds160 training,24 validation and16 separate challenge records. Easier short natural replies, grammar agreement,
short connected texts, clarifications and remembered/changing constraints complement the retained multi-turn material.
The v15 seed previously supplied separately to the owner is byte-identical to this package. It may be imported as a user dataset,
but must not replace an older installation's built-in seed without its matching manifest and application version.

All13 prior held-out files and the48-row baseline retain exact bytes. Baseline SHA256:
a1efc4e62aba0afbe0d2f062e76e8f68c8f54454c68fe985a88ece58a1d14973.
All complete sequences fit512 UTF8 byte tokens including roles/history/target. Current max is recorded in reports/dataset-audit-v15.json.
Existing workspace controls are not silently replaced. Built-in new seed merges on an explicit eligible manual training stage,
not on opening a workspace. Direct conversational creation is valid and does not need the optional minimal basic stage first.

Strict checks cover schema/role order/UTF8, lengths, exact and normalized duplicates/partition inputs, IDs/hashes and frozen
files. They do NOT establish semantic independence or model quality. Byte tokenizer remains262 IDs; expanding corpus content
is not changing embedding vocabulary or parameter count. Some ordinary topics intentionally occur across partitions.
No private logs or pretrained model are bundled. Application MIT license and upstream notices remain.
