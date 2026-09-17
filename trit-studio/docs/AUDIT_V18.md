# Trit Studio audit18: durable launch intent, batch64, parameter help and honest live accuracy

## Evidence boundary

Cumulative source revision based on owner master `7bc383e5e6b82fa81f0407e03056369bc8694457`,
trit-studio tree `5739bd8b99748cfdab83b814f0e2ec54f3f5eff0`. All Grok integration repairs are retained,
including DeviceType and held-out challenge loading from the integrated audit17. No owner repository was mutated.
Source author executes corpus/math/structural/package checks. **C# build/tests, TorchSharp runtime, Avalonia,
Windows/PowerShell/CUDA: NOT_RUN here.** dotnet is absent and direct official SDK access fails DNS in this container.
New real-runtime tests are acceptance obligations, not passes. Prior audit reports are historical.

## Findings and repairs

| ID | Finding | Source change / required real acceptance |
|---|---|---|
| A18-01 | A button labelled apply actually replaced LR/batch/length/steps/interval with a profile; no separate durable next-run intent | Rename to explicit Fill learning profile; add Save launch settings; automatic debounced save of valid edits. Profile fills only five fields and never starts training. |
| A18-02 | Ready events could overwrite edited launch fields from older committed settings | One-time hydration per workspace; repeated Ready/reconnect/stop/quality/rollback do not replace the initialized editor. Future intent lives in next-run.json, NOT trainer settings.json. |
| A18-03 | Culture-sensitive numeric input could retain an old Value behind invalid text, or silently revert on blur | ParameterNumber and strict parser accept comma/dot/exponents without thousands separators. Invalid text stays visible; CaptureRunDraft refuses it. Explicit profile/hydration replaces stale Text even if Value already equals the desired number. |
| A18-04 | Unsaved editor state could vanish on switching/closing, and creation had no independent persistence | Flush valid drafts before switch/disconnect/close. Refuse switch on invalid/failed write; explicit close-without-saving confirmation. Separate creation-draft.json under app home. No trainer/weight changes from saving fields. |
| A18-05 | Only changing UI batch limit would leave sampler, supervised assembly and validation capped at32 | Shared ResourceOptions.MaxBatchSize=64 across all relevant paths; creation/selected-model controls, estimates, sampler, full/partial validation and native tests. 65 remains rejected. |
| A18-06 | Parameter names lacked explanations | Opt-in HelpField for architecture and launch labels only, ShowDelay2000ms and BetweenShowDelay0. Bounded wrapping text, no global tooltip policy change. |
| A18-07 | No always-visible learning indicator with clear semantics | Token-weighted accuracy over last<=32 training batches, plus full control accuracy at existing evaluation boundaries. Reuse existing logits, one combined host metrics transfer; no additional forward per step. Shared header across both tabs. |
| A18-08 | Live candidate scores could be mistaken for saved progress, leak across models or remain after rejection | Counts, measurement steps and sequence lengths travel with status and committed revision metadata. Reset live windows on restore; reload saved metrics on rollback/reopen. Standalone packed files and old snapshots say not measured. Terminated trainer reverts to saved observations. |
| A18-09 | QualityProbe recalculated the training sample once more solely to measure accuracy | Reuse counts from the existing Evaluate pass. Keep all independent trained-export checks and public greedy probes. Do not relabel a small diagnostic sample as full control accuracy. |
| A18-10 | Preparing new live metrics exposed validation-cache invalidation requirements | Accuracy and loss share corpus/version success boundaries; failed evaluation cannot create cached success. Invalidate before restore, even if restore fails. |
| A18-11 | Zero-step estimate still parsed hidden invalid future batch fields in UI | Initialization estimate uses zero-step resources only; actual creation ignores hidden future training controls. Independent draft save can still report invalid future fields honestly. |
| A18-12 | Dataset needs continued modest growth rather than a claim of fluent pretraining | +64 training,12 control,12 held-out. All16 earlier held-out files and48 basics remain byte-identical. Old workspaces retain their control set. |

## Launch-state semantics

next-run.json stores future intent for ONE workspace and its architecture. Valid changes save after450ms of inactivity;
explicit Save writes immediately. Save never loads the learning profile. The profile is an explicit replacement of only
LR,batch,length,steps,publication interval, retaining device/RAM/architecture/selected data. It also saves a valid result.
Creation is separate: name, architecture, stage and future creation launch settings persist under app home.
Draft corruption is not overwritten by passive loading. The user must explicitly save corrected fields. Failed disk writes
retain current fields and the last valid file, with a visible failure; no claim that persistence worked on broken storage.
Drafts contain settings, not private messages or weights. Selected file paths are deliberately not automatically carried across models.

## Accuracy meaning and cost

Training accuracy =100*correct/total over the last at most32 selected batches (token-weighted). Predictions are from the forward
BEFORE each successful optimizer update; displayed step is that update's completion step. A changed encoded corpus or restored
session resets the live window. Changing batch/material changes the sampled population, so its trend is not a fixed benchmark.
Control accuracy uses all supervised targets in the SAME frozen workspace validation corpus, at the configured sequence budget.
It is updated at existing before/after-publication evaluations, not every training batch. The UI keeps its measurement step visible.
Padding and masked prompt/history positions are excluded; EOS is a supervised target. Byte-token accuracy is NOT character,
word/sentence accuracy, free-running fluency, generalization proof or a calibrated probability. No monotonic clamping, synthetic
progress or growing intelligence score. A real decrease is displayed. Existing loss-based publication guard stays unchanged.

Argmax/equality/counting adds computation but reuses already available logits. No second forward/softmax; GPU metrics share the
existing loss+gradient-norm host transfer. Whole-control loss and correct count transfer once at the end of evaluation. No extra
per-step disk writes: metrics persist with committed revision.json. Cancelled/unpublished observations may be lost on recovery.
QualityProbe may evaluate its own samples; those observations must NEVER replace the worker's whole-control score.

Accuracy metadata is optional/backward compatible. Existing v14-v17 snapshots have no accuracy and show not measured until a real
check. A standalone .tritmodel has no training history/metadata. Saved weights and checkpoints are not migrated/retrained by opening.
Checks validate positive denominators, count bounds, steps, sequence budget and snapshot correspondence before accepting metadata.

## Real-runtime acceptance

Run all managed/UI/worker/native tests and retained LearningSmoke, not only structural scripts. New suites cover typed comma/dot/
scientific values, invalid text+blur, same-Value profile recovery, autosave/two-model restart, repeated Ready fencing, disk failure,
independent raw creation, Batch64/partial controls, exact request/saved-options equality and second custom LR after reopen.
Metrics: masked denominator, uneven batch weighting, rolling eviction, honest decreases, cache invalidation, full/selected native
logits, actual Batch64 updates, committed metadata, restart and rollback, always-visible header and model switch clearing.
Use actual Windows pointer dwell at2sec for each intended label, successive labels included. Inspect at all supported UI scales/DPI.
Run real CPU and RTX5080 CUDA. Increasing batch can cause OOM or lower speed; there is no promise64 fits every network/context.
Do not weaken tests, add canned replies, silently change LR/cadence, change vendor versions or pack native libraries in the update.

## Residual limits

No C# compile/UI/GPU pass, measured speedup, native leak-free result or new fluency claim from author. Fixed byte tokenizer/V1
model format remain. Settings persistence is not universal power-loss durability. Local workspaces are trusted, not sandboxed.
Only last committed metric history persists, not every plotted datapoint; no plotting or cross-model ranking was introduced.
Real increases in a control score are encouraging evidence on that task, not assurance of correct conversation.
