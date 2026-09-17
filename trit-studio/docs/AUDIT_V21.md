# Trit Studio audit21: context transfer and guard-compatible conversation phases

## Baseline and evidence boundary

Full cumulative revision of the supplied audit20. Owner fork master was read again at
`68727516a4cb06bf99e971991bd95d9dd6632d4c`; app subtree `1c31d80d6dd478091f19d6fa2cc6ff58f864bc5e`.
No owner checkout or remote was changed. Preserve every integrated host repair and every earlier gate.
The binary format remains V1 and vocabulary remains 262 UTF8-byte IDs. Dependency versions did not change.

Executed: actual independent PyTorch CPU learning, exact data generation/isolation checks, source/script
checks and local Git package rehearsals. **C# build/tests, TorchSharp native runtime, Avalonia, Windows,
PowerShell and CUDA are NOT_RUN by the author.** dotnet is absent; official SDK access failed DNS.
Written native tests and independent Python comparisons are not runtime passes. Earlier audit reports
are historical. This is source for Grok, not compiled Windows output or a fluent pretrained model.

## Findings and source changes

| ID | Finding | Change and real regression obligation |
|---|---|---|
| A21-01 | A starter-only first course phase can worsen full control loss enough to trigger the unchanged publication guard before the dialogue phase. The audit20 independent full experiment omitted this guard. | Explicit ContextPractice opt-in uses all training data in EVERY phase with changing starter/context reference replay. Guard formula remains unchanged. New lab --enforce-guard returns code3 on rejection and keeps earlier exported files, not a false completed-course result. |
| A21-02 | Context supervision contained few closely related fact swaps, while generic acknowledgements can hide failure to use preceding user facts | Add388 explicitly template-labelled context dialogues: names, other speakers, corrections, moved objects, completed vs pending actions, unknown information. Apply existing all-assistant-target expansion and full/effective control isolation. These are exercises, NOT388 independent stories. |
| A21-03 | A keyword check can pass a generic answer without proving that it changes when the preceding fact changes | Separate public context-challenge:16 PAIRS/32 variants, same final question and different facts/targets. Both variants must match exact normalized facts; all previous replies are really generated. No teacher reply injection. A constant answer cannot pass a pair. |
| A21-04 | Only the first category could be tested by a bounded automatic subset | Spread4 automatic pairs across all16; explicit full check runs16. Preserve original10-case probe separately. Record actual revision/hash/step, dropped context, repetitions, full answers and expected PUBLIC facts only in the report. |
| A21-05 | Adding a new report could fail an already successful checkpoint or confuse old UI receipts | Additional probe errors emit a warning; original report and committed model remain. Cancellation remains cancellation. Receipt has optional contextSummary/contextReadable/contextError, UI tolerates old receipts and rejects stale workspace events as before. |
| A21-06 | A new training option could regress durable next-run settings or accidentally affect legacy models | ContextPractice defaultsfalse in old JSON, requires Course, is separate for creation and selected-model runs, participates in autosave/hydration. Only the explicit conversation profile opts in. Course opt-out clears the flag without erasing already learned weights or saved data. |
| A21-07 | A source-only data change could miss adjacent trainer/checks copies | Both context files are manifest/version/hash checked and app-owned update data. Only conversation-context is TRAIN. context-challenge ALWAYS uses nontraining loader. All historical heldouts, controls, starter and baseline stay byte-identical. |

## Sampling semantics

Let N be the full expanded TRAIN corpus. Each phase includes ALL N entries at least once.
- Phase0,20%steps: N full +2N starter references +N context references.
- Phase1,40%steps: N full +floor(N/2) starter +N context.
- Phase2,40%steps: N full +floor(N/4) starter +floor(N/2) context.

References reuse encoded arrays, not duplicated byte buffers. This intentionally changes sampling probabilities,
not the number of unique authored records. It is not uniform task balancing; context lengths and bucketed
minibatches still affect correlations. The original course remains selectable with ContextPractice off.
Every explicitly started course begins phases and LR schedule again. Plain continuation can turn Course off;
previously saved expanded TRAIN records remain. Nothing is inserted into raw zero-step initialization.

## Actual independent results, including the failures

All runs below use analogous Python QAT, seed42, medium821120params, B8, peakLR.001 with cosine/warmup,
new v21 data, same full408 controls. This is NOT the user's4.79M model, not C# initialization/RNG, not CUDA.

1. Controlled A/B,3000requested,publication250: both arms have the SAME new data. Old starter-only phase
   rejects step500 (control loss2.21965 after1.61932 at250), restores250 and exits. New mixed phase finishes3000,
   control loss1.16373,accuracy59.295%. Neither arm passes a paired unseen-context case. This isolates the
   guard/sampling failure, not a claim that all old runs stop at500 or that mixing alone fixes fluency.
2. Mixed profile6000,publication500: finishes with loss1.06951,accuracy64.070%. Familiar greeting/book replies
   work, but novel-context transfer is0/16 pairs (0/32 final exact facts). Open unfamiliar queries still have
   generic or incorrect responses. DO NOT call this an adequate general conversational model.
3. A supplemental known-TRAIN sample of14 context dialogues on the6000-step model yields14/14 final exact
   answers, using its OWN earlier generated replies. This demonstrates learned retrieval on known forms,
   not generalization. Sample indices and all answers are retained. It is not an exhaustive training score.
4. An alternative smaller residual initialization was tried independently: final loss1.0470,accuracy63.592%,
   still0/16 unseen pairs and poor open answers. It is NOT adopted in production. The experiment code and
   raw negative result remain for reproducibility; existing model initialization is unchanged.

See reports/context-ab-v21.json,context-full-v21.json,context-known-v21.json and
context-initialization-trial-v21.json. Training state .pt files are not silently shipped as user models;
the scripts can recreate them. No decoder patch, canned reply fallback, external language model or filters.

## Test design and limitations

The new pair suite uses distinct question/intro templates, with deliberately familiar names/colors and
subjects. It is a PUBLIC development benchmark, not blind heldout evidence. The extra earlier assistant
prefixes of its records are also checked against derived training inputs. Exact normalized matching refuses
incorrect extra sentences but may reject a correct paraphrase. Four automatic pairs are only a sample, clearly
labelled; manual16 is full. Old ten public conversational checks retain their legacy weaker semantics.

Each variant starts empty and feeds real generated answers into later context. A lost history or repetitive
reply prevents pair success. Expected facts are read solely by the evaluator after generation. The production
decoder has no special cases for the suite. Reports may expose learned private text through generated output;
inspect before sharing. Check reports are not added to training and do not change the queue or model.

## Required real gates

Run ALL earlier core/UI/worker/native tests. Add Audit21Checks and actual UI profile/opt-out/repeated Ready/
model-switch tests. Run true worker raw->restart->context-course versus direct creation parity, and read-only
report invariants (same active state/hash/replay/mode, both complete report files). Existing six-reply recall,
trained export parity, objective gradients, updates, cancellation and independent fixture remain mandatory.

Run an isolated native --learning-lab --course --context-practice --enforce-guard experiment. A rejection's
exit3 is not a compiler failure and NOT a completed-course pass. Keep evidence and do not disable the guard
to make the report green. A disclosed shorter CPU pilot is acceptable before a full laptopCUDA course;
report actual steps/device/outputs. Read novel answers; known recall and teacher accuracy are insufficient.

The default delivery is still the small win-x64 app-only UPDATE with both trainers/data, no vendor DLLs.
Actual Windows UI and GPU execution remain target gates. No altered tests, canned answers, stale ZIP or
claim of conversational acceptance merely because the build or optimizer completed.
