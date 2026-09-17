# conversation-ru-v18

Original synthetic Russian starter material, MIT. No downloaded private chats, scraped text or user weights.

2826 training,388 validation,292 held-out records across versioned splits;48 separate frozen basic texts. Total3554.
This audit adds64 training (40single-turn,16multi-turn,8texts),12control and12held-out.
Natural brief conversation, keeping constraints/preferences, careful clarification and short rewrites. See tools/corpus_v18_additions.json.
All16 prior blind split files and basic pretrain remain byte-identical. Existing workspace controls are NOT automatically replaced.
Full records fit the default512-byte-token training budget. Exact measured maximum and hashes: reports/dataset-audit-v18.json.
The262-ID byte tokenizer is unchanged, not an expanded neural vocabulary. This is data, not pretrained weights or a fluency result.
The independent six-reply learned model is an explicit test fixture, never stock knowledge for new models.
