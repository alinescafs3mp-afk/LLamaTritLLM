# Trit Studio audit19: reproduce degeneration before promising conversation

## Baseline and evidence boundary

Owner fork: alinescafs3mp-afk/LLamaTritLLM, master, supplied deployed commit
68727516a4cb06bf99e971991bd95d9dd6632d4c. Exact trit-studio tree reconstructed locally:
1c31d80d6dd478091f19d6fa2cc6ff58f864bc5e. Retain Grok's headless tooltip overlay, TranslatePoint,
kvHeads fixture reset and all earlier host repairs. No owner checkout/remote was modified.

Executed: independent PyTorch CPU learning of the ACTUAL v18 corpus with a4,788,992-parameter analogous
ternary architecture, corpus-only bigram baseline, Python algorithms, structural checks and synthetic Git
package reproduction. C# build/tests, real Avalonia, TorchSharp, Windows, PowerShell and CUDA NOT_RUN here.
No dotnet compiler is available. The reported Python results are not results from the user's weights.

## Findings and implementation

| ID | Finding | Change and required native gate |
|---|---|---|
| A19-01 | LR.01 remained constant throughout manual learning; it can enter a low-information repetitive regime | Independent same-initialization A/B experiment reproduces the owner's specific symptom. Add explicit optional warmup/cosine plan; legacy jobs remain constant. Show actual last LR and persist it with state. Native optimizer group-rate/resume test required. |
| A19-02 | .01 was mistaken for a minimum and .1 seemed a logical escalation | Make upper-bound error unambiguous; high-LR warning/confirmation above.003. Bound remains.01. Threshold is a caution, not a universal mathematical stability limit. |
| A19-03 | Raw creation attached48 not-yet-trained baseline rows; first conversation later retained them, while direct creation used seed only | Only at optimizerStep0, first conversation with bundled material removes exact placeholder baseline IDs. Real direct vs raw->close->open->train worker test compares corpus hashes and updated weights. Positive-step basic knowledge/material is NOT removed. |
| A19-04 |128-snapshot count refused an otherwise possible long job | Explicit opt-in cadence planner fits at most32 new snapshots or the available slots, never speeds up publication or deletes files. Supports up to100000-step explicit jobs. Existing strict mode/late disk checks stay. Disclose effective interval and potential loss on crash. |
| A19-05 |42% byte accuracy can be reached without a language model | QualityProbe now fits a byte-bigram on TRAIN targets only and evaluates FULL unchanged workspace controls. Also returns repetition diagnostics and exact-known-target match on three public prompts. No answer substitution or controlled-data training. |
| A19-06 | Six-reply recall did not measure learning on the real corpus | Add opt-in native --learning-lab with real experimental architecture, checked bundled corpus, optimizer updates, disk export/CPU autoregression every100steps, byte-reference and saved artifacts. Completed experiment is NOT a fluency pass. Keep six-reply and independent fixture gates. |
| A19-07 | Repetitive output was labelled only a successfully completed generation | Add descriptive repetition notice alongside unchanged output. Same notice for reloaded turns. No stop/rerank/language mask or hidden reply replacement. |
| A19-08 | New controls could repeat earlier draft-loss bugs | Schedule/cadence controls participate in per-model and independent creation draft save/hydration. Profile explicitly enables both; Save never does. Real UI tests include custom opt-outs and repeated Ready/reopen. |

## Executed learning investigation, NOT a native acceptance result

Both arms use the same independent Python initialization, RNG recipe, corpus and batch16,2 CPU threads.
Architecture: width256,FFN768,layers6,heads8,KV4,planes2,group32,threshold.5. Same as the screenshot's
architecture, but NOT C# initialization, NOT batch64, NOT owner's saved weights. Each arm ran300updates.

| LR | Control loss at300 | Control byte accuracy | Three free responses |
|---|---:|---:|---|
|.01 constant|1.792526|42.1424%|Repeated Cyrillic o on all3 prompts|
|.001 constant|1.299044|53.3978%|Russian-like fragments, still NOT coherent conversation|

At100 updates the.01 arm had42.1977% and already reproduced the o-loop. The lower-LR arm did NOT solve
fluency at300steps. No measured benefit of the NEW cosine schedule is claimed from these constant-LR arms.
The independent helper stops after72bytes and can replacement-decode a final partial scalar; that is NOT
evidence against the production UTF8-safe decoder. Preserve the raw experiment outputs and scope.

Byte-bigram fitted only to v18 seed+48basic TRAIN targets: controls9440/21705=43.4923%, without any neural
network. It is not a conversational system. Training plus future baseline occupies533514 encoded input
byte-token positions in one pass, excluding JSON metadata. Repeating a small corpus is not broad pretraining.
See reports/learning-investigation-v19.json and reproducible tools/learning_corpus_experiment.py.

## Semantics and limits

- Warmup/cosine is OFF for legacy saved plans. Profile enables it visibly. Per explicit new manual job:
  min100 or floor(steps/10) warmup updates, peak then cosine to10%peak. Short jobs can have zero warmup.
  Every NEW requested job starts a NEW curve. Interrupted jobs restore weights/optimizer/last actual LR,
  not an automatically resumed pending job/cosine position. No silent automatic LR reduction or retries.
- The old validation regression guard is unchanged. A large enough degradation still cancels the candidate
  and restores the last committed snapshot, not necessarily the globally best checkpoint. No early-stop
  event certifies that the retained model is a useful chatbot. Wrong training may still be checkpointed.
- Auto cadence only changes this explicitly requested job, not next-run intent. Existing51/8000/100 becomes
  interval250 with32 new snapshots; an already requested1000 stays1000.128 remains a finite storage guard.
  When no slot remains, explicit maintenance is necessary. No rolling deletion, disk reservation or indefinite-run claim.
  Wider intervals delay validation and can discard more unfinished work on failure/cancel; the UI warns.
- Raw->conversation correction applies at step0 only. Existing trained models retain all old data. It
  removes an incidental data/order difference but is NOT proven to explain all differences in the screenshots.
- Baseline uses frozen workspace controls, not the latest bundled file if the workspace has older controls.
  It counts masked teacher-forced byte targets like the network; it is not semantic accuracy. QualityProbe's
  small sample must not replace full-control UI observations. Repetition thresholds are heuristics.
- Reports contain fixed public prompts and generated text; generated text MAY reveal learned private data.
  No raw user chats or expected private dataset answers are copied. Inspect reports before sharing.
- New baseline preparation only runs in explicit diagnosis, not on every training update. Inference stays
  independent of CUDA. Library versions,262-ID tokenizer, ternary V1 format and user files are unchanged.

## Required real acceptance

Run all retained tests, Audit19Checks, expanded draft UI tests, real route-parity worker test and native
LearningPlanSelfTest on CPU/CUDA. Keep exact learning/export six-reply gates, trained prefix parity,
optimizer resume, clear/private chat, model identity, and ALL update-payload/vendor exclusion checks.

Then run an ISOLATED corpus learning experiment on the Grok host at.001 for at least300steps (more if
needed), report the actual generated text and baseline, and inspect both routes with identical settings.
A completed learning lab MUST NOT be advertised as coherent conversation. If failure is numerical or
native, fix it and repeat. If it is poor learning, preserve evidence and state it, not canned replies.
The original CPU+CUDA app-only delivery remains. No vendor upgrade, no giant full ZIP as fallback.
