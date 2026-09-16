# Audit14 · 2026-09-16

- Prepare/verify a destination workspace before disconnecting the usable model; release rejected lease and reuse verified objects.
- Avoid the initial native-to-host master round trip only while random weights remain untouched; invalidate before mutation/restore.
- Portable vector RMSNorm scaling/residual addition with tails, aliases, shape/finite/cancellation guards.
- Observe pre-cancellation before opening model/hash/revision files.
- Add real native/headless/managed regressions and initializationAndElementwise benchmark; NOT_RUN in authoring environment.
- Dataset3090 records including2474 training; keep baseline and all earlier holdouts frozen. Full max453.
- Host integration: TorchSharp 0.107 Tensor.mean takes long[] dimensions; restore the same reduction axes.
- Host integration: broken trainer protocol frames fail the pending request as IOException (same as EOF) and still kill the child.
- Host integration: EOF fixture closes every Linux duplicate of redirected stdout so .NET 10 actually ends the parent pipe.

## Historical changes

# Audit13 · 13.0.0-audit13 · 2026-09-16

- Capture journal EOF once, bounded short-read loop, newest-first parse and strict UTF8/captured-context validation.
- Restore failed/cancelled excluded prompts with exclusion; preserve new drafts and their flags.
- Portable SIMD weighted-value attention with scalar fallback, cancellation and shape/alias guards.
- Add22 managed contracts, real headless private retry checks and actual attentionAndJournal benchmark.
- Corpus:2314 seed,304 controls,224 held-out,48 frozen basics =2890. No pretrained weights.
- Source author did not execute C# build/tests,actual GUI/native/CUDA. See docs/AUDIT_V13.md.

# Audit12

- Group CPU Q/K/V and FFN gate/up schedules, closure-free tiny sequential path, cached RoPE denominators.
- Reuse identical full/effective split keys; preserve legacy identity and cancellation.
- Cancel superseded snapshot loads, await identical notifications, retain workspace lease until reads drain.
- Validate command name and duplicate event/terminal fields before completing correlated requests.
- Add272 total managed contracts, real malformed child modes, headless supersession test, managed benchmark.
- Add200 original data records; baseline and earlier blind files remain frozen. Native tests not run by author.

# 11.0.0-audit11 · 2026-09-16

Selected-only online replay encoding with legacy pool parity; conservative immutable corpus reuse;
strict incoming command envelopes; workspace-bound deferred learning-rate edits and confirmed mode feedback;
observe abandoned graceful-shutdown write faults; preflight the exact worst-case publication count before training.
Managed preparation benchmark and 251 defined managed checks plus real worker/headless cases.
No C# build/native/UI/CUDA execution is claimed in the authoring environment.
Dataset:1994 seed +256 controls +192 holdout +48 frozen basics =2490. See docs/AUDIT_V11.md.

# 10.0.0-audit10 · 2026-09-16

Exact final-buffer UTF8/token preparation and small-pool sequential encoding; bounded outgoing command transfer without
training-completion deadline; fail-closed in-memory mode after persistence refusal; mutation-aware recovery avoids full
GPU/native reload on rejected preflight. Native preparation benchmark,218 defined managed contracts and two new child
transport scenarios plus actual-worker storage-fault/refusal checks. C#/native/UI/CUDA NOT_RUN by source author.
Dataset:1834 seed +232 controls +176 holdout +48 frozen basics =2290. See docs/AUDIT_V10.md.

# 9.0.0-audit9 · 2026-09-16

Historical summary: bounded checkpoint corpus JSON cache, strict bounded UTF8 imports, early snapshot counter consistency,
prepared batch metadata, cancellable exact changed-weight counts, separate completed-answer/journal-save failure feedback.
2090 corpus records,202 managed cases defined. Authoring evidence was independent Python/structure/packaging only.

# 8.0.0-audit8 · 2026-09-16

Terminal completion parsed before request removal; rollback restore before active pointer; bounded CPU validation preparation reuse;
no whole-ledger replay encoding for manual training; durable actual-material provenance and accurate immutable-stage receipts;
sequence-budget-aware historical validation scores; bounded cancellable bundled manifest/data loading.
Dataset:1514 seed +184 controls +144 held-out +48 frozen baseline.172 managed tests defined, not executed by source author.
See docs/AUDIT_V8.md and docs/PERFORMANCE_V8.md for evidence limits and real acceptance.

# Audit7

Explicit random/basic/conversation stages and preserved inference-only references; online off for stage comparison;
fixed captured teaching context; native SiLU; bounded generation bytes and allocation-free reuse of validated sampler settings;
UI version/caption drift fixed;160 new conversation examples,24 controls,16 blind cases,48 separate baseline texts.
141 managed tests are defined but not executed by the source author. See docs/AUDIT_V7.md.

# 6.0.0-audit6 · 2026-09-16

Performance: group-local pack, parallel unpack, hash-on-write snapshots, cancellable publication preparation, tensor-local staging,
scoped gradient norm, QAT magnitude reuse, prebound CPU layers, no unused last forward and linear greedy selection.
Correctness: effective-context split isolation, old replay recheck, error/cancel bubble feedback, corpus/app/test version coherence.
Data:1194 train/136 control/112 held-out. Source only; native/desktop/CUDA gates await Grok and target execution.

# 5.0.0-audit5

Performance/correctness source audit: supervised-only final projection with full-output comparison path; scoped in-place QAT scratch;
one-upload restore; bounded online replay sampling; shared CPU RoPE and response-local sampling/logits buffers; cheap context budgeting.
Incremental bounded protocol input, fatal-child termination, answer-independent validation guard, bounded JSON staging, UTF-8 budget handling.
90 defined managed contracts, expanded native parity/child/UI gates and three-way synchronized native benchmark. C# NOT RUN here.
conversation-ru-v5: 970 train +112 control +40/24/16 frozen holdouts +16 new challenges =1178. No pretrained weights.

# 4.0.0-audit4

Performance-focused source cumulative: scoped QAT/evaluation caches, bucketed batches, optional SDPA, packed-only snapshot activation,
committed master reuse, one validation scalar read, throttled progress/latest-only streaming, replay summaries and stricter numerical/state/export guards.
64 managed contracts defined, expanded native and actual child protocol regression tests. Native isolated benchmark in host/target delivery.
Corpus conversation-ru-v4: 746 train +88 control +40/24 preserved blind generations +16 new challenge =914. Old tests unchanged.
C# compiler/native/UI/CUDA were NOT RUN by source author; all delivery gates remain mandatory.

# Audit 3 · 2026-09-16

Version 3.0.0-audit3. See docs/AUDIT_V3.md for source changes and evidence boundary.
Added live context and completion reasons; explicit queue/snapshot maintenance; monotonic revision IDs;
chat crash-tail recovery and bounded history; persisted online LR; cancellable discovery and idempotent child shutdown;
aggregate import guards; bundled corpus verification and seed upgrade for existing models; expanded runtime contracts.
Corpus grew from 378 to 650 records, including 522 training. Old blind test bytes are unchanged.
C# build/UI/GPU not run in source authoring environment.

# Changelog

## 2.0.0-audit2 · 2026-09-16

Cumulative delivery for the user's fork, preserving original LLamaTritLLM source.
Adds the C# portable delivery orchestrator, target diagnostics and exact artifact receipts.
Repairs coherent checkpoints, optimizer device restore, cancel-before-start fencing, worker receipts,
UI feedback/state synchronization, memory lifecycle and sequence/data handling.
Adds 298/40/40 original conversational records, strict multi-turn training and split protection.
Adds core, headless UI and real-worker acceptance tests, with unexecuted C# status explicit.

## 0.1 / audit1

Initial C# desktop source, byte tokenizer, independent CPU inference and TorchSharp training.
Historical source-only delivery; no compiled release was certified.

## 9.0.0-audit9

Bounded immutable corpus serialization cache and hash-on-write for checkpoints; exact cached batch statistics;
parallel/cancellable weight-change telemetry; strict bounded UTF8 TXT/JSONL imports; metadata/state counter and committed-ID
agreement before restore; distinguish completed generation from failed chat-journal append. Add corpus-v9 and202 managed
contracts in total. C# runtime gates remain NOT_RUN by the source author. See docs/AUDIT_V9.md and docs/PERFORMANCE_V9.md.
