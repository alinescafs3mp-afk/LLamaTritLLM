# conversation-ru-v14 dataset card

## Original synthetic starter material

Reviewed Russian examples written for this tiny local transformer project, not scraped text, user chats, pretrained weights or
factual-expertise evidence. Deterministic source modules and tools/build_conversation_dataset.py reproduce the JSONL files.

| Split | Records | Intended use |
|---|---:|---|
| seed.jsonl | 2474 | 1498 single-turn,672 multi-turn,304 plain texts; actual training |
| validation.jsonl | 328 | Repeated candidate guard, not a blind generalization score |
| test.jsonl and challenge*.jsonl | 240 | Held-out prompts, no ordinary training import |
| pretrain.jsonl | 48 | Separate frozen basic pretrain for the before/after experiment |
| Total | 3090 | Not all records enter training |

Audit14 adds96 single-turn,48 multi-turn,16 text,24 control and16 challenge records. Topics include low-pressure everyday company,
precise clarification without guessing, distinguishing rarely/never, intended/started/completed, ownership and changing decisions,
short rewrites, exact one/two-line output constraints and small creative tasks. Some targets contain actual newline characters.
A source-only escaping fix is scoped to v14 additions; previous corpus bytes are not rewritten.

The48 baseline texts and all twelve previous holdout files are byte-frozen. New challenge-v14 has16 additional records. Existing
workspaces keep their old validation set. Updated training rows merge only on explicit manual training with bundled updates on,
not on opening a workspace. No baseline stage silently consumes conversational/custom/replay data. First-stage references remain
inference-only, immutable and separate from resumable master/optimizer snapshots.

## Checks and limits

Maximum COMPLETE sequence453 UTF8 byte tokens, including roles/history/answer/EOS, fits default512 without trimming. Previous
maximum437 has increased; don't carry stale length claims into release notes. Validation checks strict schema/UTF8/roles, exact
and normalized identities/input-history partition overlap, SHA256/counts/lengths and all frozen files. Equivalent single-turn and
messages schemas are normalized. These checks do NOT prove semantic independence: everyday topics can overlap across partitions.
Control scores are repeatedly consulted. Inspecting test outputs can also compromise a blind experiment; retain a genuinely new
private evaluation set for final judgments. Tests may not be merged into training to improve a score.

Frozen baseline SHA256:a1efc4e62aba0afbe0d2f062e76e8f68c8f54454c68fe985a88ece58a1d14973.
Starter data is not enough to establish general-language competence or fluent conversation. Measure after real training on novel
prompts. Ordinary learning changes weight values, not the parameter count. No trained model is bundled. All example content is
provided under the application license; upstream MIT notices remain with the reused project material.
