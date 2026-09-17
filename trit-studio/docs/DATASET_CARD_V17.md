# conversation-ru-v17

2762 training records,376 validation records,280 held-out records,48 separate frozen basic texts:3466 total.
Increment:32 training,8 validation,8 held-out. Small audited increment intentionally prioritizes engine diagnosis.
Short natural Russian replies, agreement, concise rewriting, explicit unknowns and small multi-turn distinctions.
Original synthetic content, no scraped text or private conversations. License MIT, matching the source project.
All15 prior held-out files and48 baseline texts byte-frozen; full sequences fit512 byte tokens.
The tokenizer remains262 IDs. A larger corpus is not a larger embedding vocabulary or evidence of fluent conversation.
Existing workspace validation is preserved. New training rows are merged only through the explicit fine-tuning route.

The separate learning-reference.tritmodel fixture is NOT part of this corpus, not an automatic pretrain and not a general assistant.
It memorizes6 short replies through real Python gradient updates and exists to isolate the C# decoder from training.
