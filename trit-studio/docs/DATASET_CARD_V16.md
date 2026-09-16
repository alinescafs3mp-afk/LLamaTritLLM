# conversation-ru-v16

Original synthetic Russian starter corpus, MIT, no scraped text or owner chat logs. Not pretrained weights or a fluency benchmark.
2730 train (1626 one-turn,736 multi-turn,368 plain-text),368 control,272 held-out and48 separate baseline texts:3418 records.
This audit adds96 train (48 one-turn,32 conversations,16 texts),16 control and16 new held-out records.
The smaller increment is intentional: this pass prioritizes audit over count growth. Material focuses on short natural replies,
agreement, concise rewriting, not inventing unknown details, changing plans and tracking who/what a fact refers to.

All14 earlier held-out files and the48-row baseline are byte-frozen. Existing workspace controls remain unchanged. Every complete
bundled sequence fits512 byte tokens; maximum453. Training on the complete example preserves its final target and history.
A fixed byte vocabulary remains262 IDs. Corpus growth does not enlarge tokenizer vocabulary or automatically add parameters.

Generation: python3 tools/build_conversation_dataset.py. Validation: python3 tools/validate_dataset.py.
Checks cover schema/UTF8/hashes/exact-normalized duplicates/runtime input equivalence/length; NOT all semantic overlap,
linguistic expertise, human preference or guaranteed transfer to unfamiliar questions. Related ordinary topics occur across splits.
Do not add held-out/control records to training, approve them as live feedback, or silently replace old controls to improve scores.
