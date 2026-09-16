# Dataset audit14 · 2026-09-16

+160 seed,+24 validation,+16 new challenge; now2474/328/240/48, total3090. Training breakdown1498 single,672 multi,304 texts.
Frozen48 basic texts and twelve previous blind files remain byte-identical. New full-sequence maximum453, all fit512.
Adds precise everyday discussion, clarification, changed constraints and explicit action status. Target-format newline cases
are decoded only for v14 source additions. No automatic validation replacement or training on held-out examples.

## Historical revisions below

# conversation-ru-v13

Add160 training examples(96 single-turn,48 multi-turn,16 text),24 controls,16 new challenge-v13 records.
Totals:2314/304/224 held-out/48 basic =2890. Preserve all eleven earlier holdout files and the48-text basic corpus byte-for-byte.
More everyday context, explicit constraints, changing plans, entity distinctions and concise requested rewrites.
No private conversations or downloaded pretrained models. Normalized exact isolation is not semantic-independence proof.

# conversation-ru-v12

+160 train (96 single-turn,48 multi-turn,16 plain),+24 controls,+16 new challenge records.
Total2690:2154 seed,280 controls,208 held-out,48 basics. Frozen baseline and ten prior blind files unchanged.
Expanded low-pressure conversation, accumulated constraints, separate persons, concise formats and honest unknowns.
All complete sequences <=437 byte tokens, fit default512. No new pretrained weights or measured quality claim.

# conversation-ru-v11 · 2026-09-16

Add160 authored training examples:96 single-turn,48 multi-turn,16 plain texts. Add24 controls and16 new
challenge-v11 examples. Total2490:1994 seed,256 control,192 held-out,48 frozen basics.
All nine prior blind files and the48-row basic pretrain remain byte-identical. All full sequences fit512.
Topics: undemanding companionship, listening without unsolicited advice, short rewrites, precise output constraints,
continuity of facts and preferences, corrections and updated decisions. No private data or pretrained weights.

# conversation-ru-v10 · 2026-09-16

Added160 training examples (96 single-turn,48 multi-turn,16 texts),24 controls,16 new challenge-v10 cases.
Totals:1834 seed,232 controls,176 holdouts,48 frozen basics =2290. Eight old blind files and the baseline are unchanged.
New coverage: low-pressure chat, context/state continuity, evolving preferences, restrained clarifying questions,
short rewrites, exact formatting, small stories. No private data, pretrained weights or fluency claim.

# Data evolution

## v8
+160 conversation training, +24 control, +16 held-out. Totals:1514/184/144 plus the SAME48 basic records =1890.
96 new single-turn,48 multi-turn and16 plain texts; baseline and all six previous blind files byte-frozen.
Explicit frozen-baseline and prior-challenge hashes added. Largest full sequence437, default512 preserved.


## v7
+160 conversation training,+24 control,+16 new blind challenge;48 separately selected baseline texts.
Totals:1354 train,160 validation,128 held-out,48 baseline. Prior v2/v3/v4/v5/v6 blind files remain byte-frozen.
Previous conversation train rows are preserved by the deterministic builder. Baseline is not silently inserted into fast conversation mode.

## Earlier increments
v2:298/40/40. v3:522/64 plus preserved/new holdouts. v4:746/88. v5:970/112. v6:1194/136.
See tools/corpus_v2.py and corpus_v3..v8_additions.py for original authored increments and DATASET_MANIFEST.json for hashes.

## conversation-ru-v9

Original additions:96 single-turn answers,48 multi-turn conversations,16 short text records,24 controls,16 separate held-out challenges.
Totals:1674 seed,208 control,160 held-out,48 unchanged baseline =2090 records. All7 prior holdout files and pretrain.jsonl are byte-frozen.
Coverage: unhurried everyday conversation, listening without unsolicited advice, cumulative constraints, updated decisions,
reference resolution, concise rewriting, explicit uncertainty and drafting without claiming external action.
Validation now also normalizes runtime input identity across supported JSON schemas. No pretrained weights or fluency claim.
