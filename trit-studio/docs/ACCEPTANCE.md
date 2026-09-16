# Audit16 current acceptance additions

Current source16.0.0-audit16, corpus conversation-ru-v16. All earlier suites below remain mandatory; historical counts are not
current passes. Author executed Python/source/package checks only. C#, native TorchSharp, UI, Windows/CUDA and PowerShell are NOT_RUN.

- Run all372 managed contracts plus retained actual-child, headless UI, worker and native suites. Audit16UiChecks must use real
  focus and KeyPress/KeyRelease: Shift+Enter inserts a newline, plain Enter sends once, a held Enter does not send a new draft,
  unavailable Send cannot be bypassed. Test IME composition and modifiers manually on Windows; headless events do not certify IME.
- A destination with missing active.json but existing r* revision folders, directory-valued active.json/chat-state.json or invalid
  bounded metadata must refuse BEFORE disconnecting the old model. Preserve epoch, client, weights, history, draft/privacy.
  Check the failed destination lease is released; preserve actual empty and staging-only creation. No guessed repair.
- Exercise two quick model-removal clicks and one cancelled confirmation. No duplicate modal or swallowed exception. Force
  trash failure and verify error feedback remains; if active-session recovery succeeds, restore its original draft/privacy.
- Force a first failed CUDA probe followed by a successful explicit reconnect on suitable hardware. No cached permanent CPU
  fallback. Positive cache is permitted but cannot certify the runtime after a later driver/device failure.
- PrefixFinalBlockSkips must be n-1 only for optimized prefill. Compare all logits and continued tokens to optimizePrefix:false
  for one/multiple layers and GQA. Check cancellation and invalid input. Retain trained native->master->packed->CPU parity
  after real optimizer steps, not random-model parity alone. Run CPU and actual CUDA native suites.
- Run prefixPreparation benchmark with both orders and every earlier benchmark. No mandatory speed ratio. Record runtime/CPU
  and allocator behavior. Retain repeated create/train/evaluate/reopen soak and prior checkpoint/online privacy guarantees.
- App-only UPDATE is win-x64 and contains no vendor runtimes. Both trainers/current adjacent data are included. Linux target
  with default update refuses before builds; do not silently use --full. Run Test-LaptopChecks.ps1, which extracts the ACTUAL
  Run-Check body and tests exit0 with stderr and exit7. Log-write failures and nonzero exits must still fail checks.
- Dataset3418=2730train+368control+272held-out+48baseline. Baseline and all14 previous blind files must be byte-identical.
  Whole sequence maximum453 fits512. Existing workspace controls remain unchanged. No fluent-language claim from these counts.

Host gates must pass before emitting the fresh small TritStudio-win-x64-update.zip. Retained vendor hashes are checked on a
disposable copy of the compatible installed folder. Windows desktop and actual RTX5080 acceptance remain target-side gates.
A CPU fallback is never a CUDA pass. No weakened assertions, unreported native failure, stale ZIP or modified user workspace as fixture.

## Accumulated earlier acceptance requirements

# Audit15 current acceptance additions

Current delivery overrides historical portable wording below: DEFAULT is app-only Windows UPDATE, not the full vendor-runtime ZIP.
Run all retained tests and the current count in static-audit-v15.json. Validate trained-export parity after actual updates,
Audit15UiChecks modal clear/cancel/failed-state/reload/no-weight-change, active file selection and inactive removal. Test
model catalog refresh/failure, early switch refusal, learning target identity, lazy feedback expansion, scale/DPI and blocked
operations during switch/clear/delete. Observe actual UI, no static-layout pass substitution.

Verify every update entry with UpdatePayloadPolicy. Both trainer assemblies, adjacent datasets and checks/reference.json update.
Vendor DLLs/.NET are ABSENT from update but present and exact in installed folder. Missing/mismatched vendor fails Check-Update.
Merge onto a disposable COPY of an existing v14 full installation and run actual CPU/CUDA/desktop tests. Do not mutate user
workspaces in tests. Restore prior folder copy on failed update. Full packaging only explicit --full by owner request.

All previous numerical, protocol, lifecycle, staged training and memory-soak gates remain below as historical accumulated tests.
Source-author reports are not C# test passes. No assertion weakening to fabricate success.

# Audit14 current acceptance: retain every previous gate

Current314 managed contracts plus ALL retained actual-worker, headless and native suites. Historical headings and test/data counts
below describe the versions in which checks were introduced; they are not today's pass counts. None ran in the authoring container.

1. WorkspacePreview must be read-only and bounded; bad chat-state/active pointer/hash must refuse before old-client teardown.
   Real headless window must preserve model reference,epoch,history,draft and release only the failed destination lease.
   Valid open uses preview arrays, native restore still fully verifies optimizer. Test same-folder reopen and closing during preparation.
2. Exact initial-reference identity and native FP32 equality, precancel, Model.MarkUpdated invalidation, successful/failed RestoreState,
   and fresh capture after real gradient update. No initial cache after partial native mutation. Full optimizer/evolution checks remain.
3. Norm/residual scalar parity, vector tails, whole-array aliases, no writes on invalid shapes or pre-cancel, nonfinite reductions.
   Run full-model managed/native logits test on CPU AND real CUDA. Same reduction/multiply order is not a universal bit-equality promise.
4. Run initializationAndElementwise and ALL prior benchmarks both orders, real allocator soak, then Windows target checklist.
5. Verify3090 corpus records:2474 seed,328 controls,240 heldout,48 basics; max453 fits512. All twelve older holdouts/basics byte-frozen.

# Audit11 mandatory acceptance

Source author did NOT compile/run C#, GUI, native TorchSharp, Windows or CUDA. The 272 defined managed cases are not passed cases.
Run scripts/deliver.sh with every existing host gate. Do not weaken assertions or replace the model with canned responses.

1. Preserve zero-step random creation, separate frozen48-text basic stage, explicit conversation training, manual-stage online off and replay safeguards.
   First conversation request with bundled data disabled must fail before mutation unless known prior conversation training exists.
   First custom stage without selected data must fail unless known custom-training provenance exists. Legacy nullable provenance remains loadable,
   but old LastMaterial by itself cannot certify the material used. No selected custom files silently disappear in basic/random UI modes.
2. Preserve stage references after further training/pruning. Existing receipt must name the ORIGINAL archived revision/step, not the current model.
   Reject newly written zero-stage refs with nonzero steps and learned-stage refs with zero steps; damaged first refs warn, never get silently overwritten.
3. Run six broken-child modes. Malformed terminal error/success payloads with the CORRECT request ID must fail the awaiting request and kill the child
   before Dispose. This specifically guards the old remove-before-parse lost-completion bug. Retain EOF, oversized-line and cancellation tests.
4. Inject hash-valid semantically invalid parent trainer state. Failed rollback must leave active.json unchanged, restore current session and keep online off.
   Successful rollback publishes exactly the restored revision. A replay persistence failure after activation is a warning requiring repair, not fake undo.
5. Validation preparation: exact arrays/positions/token coverage, zero/tiny/16MiB retention budgets, cancellation on hits and before construction,
   invalid IDs not cached, short last batch. Real weight updates MUST run validation again while preparation hits increase.
   Replacing the corpus or resources resets preparation. Historical bests cannot be compared across different sequence budgets or unknown legacy metadata.
6. Bundled manifest >64KiB and pre-cancelled loads fail visibly. All previous JSON/protocol/UTF-8/snapshot/hash/lease/privacy-target guards remain.
7. Run native math/optimizer roundtrip and repeated train/eval/publish/rollback/close/reopen soak, benchmark with both cache modes and both orders.
   Capture actual RSS/CPU/time; do not invent VRAM metrics. CPU inference remains usable independently of the trainer.
8. On the target laptop run Check-laptop.cmd, then manual desktop action feedback, cancelled selections, failed material actions, before/after chat,
   restored working model after failed rollback and non-overwritten stage copies. CPU fallback is not a GPU pass.

No Windows ZIP is delivered unless it was freshly produced after all host gates, checksum matches and DELIVERY_RESULT.json says succeeded.
Always separate build-host evidence from still-pending target Windows/RTX5080 evidence.

## Retained audit9 gates

9. CheckpointJsonCache: zero/tiny/4MiB slots, exact JsonSerializer byte/SHA parity, different owner invalidation, oversized one-pass serialization,
   cancellation/failure removes only the newly created partial; existing destination stays byte-identical. Repeated snapshot restore must still rehash
   complete corpus files. Record per-slot counters and both-order `checkpointCorpus` benchmark. Test immutable-array ownership, not in-place mutation.
10. Reject trainer Step/TargetTokens disagreeing with hashed revision metadata BEFORE replacing native state. Positive mismatches, impossible zero-stage
    traces, malformed/duplicate ledger IDs and infeasible changed counts must fail. Real failed-rollback fixture must keep current active.json unchanged.
11. TXT/JSONL: valid Russian+emoji and optional UTF8 BOM succeed; malformed UTF8 and UTF16/32 BOM fail with file context. Oversized unterminated
    lines fail during bounded accumulation. Cancelled JSONL is cancellation, not a wrapped format error. Existing data remains intact.
12. Exact BatchPlanner choices/RNG state/padding/target count match the legacy oracle, including forced online examples and uneven buckets.
    CountChanged matches sequential exact counts at all thread settings; cancellation and config mismatches fail before costly work.
13. Real headless generation with an unwritable chat journal: completed answer remains, input is not restored, visible journal warning appears,
    no claim of durable chat save. Verify manually that subsequent operations can continue; unsaved turns are only in window memory.

## Audit11 gates

14. ByteTokenizer.EncodeInto: exact legacy bytes across Unicode scalar boundaries, empty/invalid input, pool boundary,
    destination capacity and offset. Invalid UTF16/short destination must not alter destination. Check Plan and Measure agree.
    Dataset.Encode must match legacy input/label arrays for EVERY partition and retained-history budget. Preserve answer/EOS masks; no silent target trimming.
    Test bounded sequential/chunked parallel EncodeAll equivalence and cancellation on empty/tiny/large pools.
15. Run a real child that emits Ready but never reads stdin. Send a legal 900000-character command; bound transfer, fail request,
    kill child BEFORE Dispose. Separately a valid child replies after 2 seconds to a small command: it MUST succeed with a 750ms
    transport deadline. These test timeouts differ deliberately; long training has no new completion deadline.
16. Force runtime.json write failure with a directory at its path. Mode enable must fail, status must say paused/off, active
    weights and queued count must remain unchanged while the worker is alive. Restore storage explicitly in the fixture.
    Old persisted flags may remain on disk; no test may pretend persistence succeeded when it did not.
17. Refuse first conversation training without required material. Count actual published events: no new event or device restore
    should occur. Existing tests of failures AFTER mutation still require complete checkpoint recovery, including sampler/optimizer/corpus.
18. Execute native `textPreparation` benchmark in both orders and record runtime allocations/timing. No required speed ratio.
    Validate manifest/frozen 48-row baseline/nine old held-out files and new challenge-v11 role/UTF8/isolation/length contracts.

## Audit11 additional gates

17. Compare old/new online replay for base-only, learned-only, wraparound and long.MaxValue step; exact input/label arrays,
    unchanged selection order and no RNG calls. Selected learned/control collisions and overlong answers must fail;
    unused legacy rows must not be encoded. At most four selected learned records per online pool.
18. Reuse prior corpus owners ONLY when ordered record values are unchanged, including Source/history ownership;
    changed data/budget still re-encodes and every real weight update still invalidates evaluation results.
19. Run the actual worker with malformed JSON, absent/typed IDs, duplicate IDs and scalar payload. Process exits nonzero
    before parent disposal. A well-formed unknown command returns a correlated refusal without killing the worker.
20. Real headless child: delayed mode edit captured in old epoch sends no command after workspace change. Navigation
    cancels pending timers; while a mode receipt is pending, incompatible model controls are disabled. Create's explicit
    opt-in online preference still works. Shutdown write tasks remain observed after wait timeout.
21. Quota preflight: exact capacity accepted, excess steps refused BEFORE new weights/publications. Preserve all snapshots;
    no automatic deletion/fewer saves. A 100000-step cancellation test uses a valid publication interval1000.
22. Native benchmark onlineReplayPreparation compares both orders, parity outside timing, warmup excluded, thread allocation
    traffic recorded. No speed ratio is a pass condition. Run the whole prior suite and target CPU/CUDA tests too.

## Audit12 mandatory additions
- All272 managed contracts, real10 broken output modes and all prior worker/native/UI gates.
- Separate/fused projection bit parity, large and tiny matrices, GQA, uneven rows, aliases and cancellation before mutation.
- Exact legacy split-key hashes and full/effective decisions, changed answers, trimmed history and cancellation during enumeration.
- Headless latest load cancels obsolete pending work; duplicate valid publication waits for same activation; fault/cancel lifecycle owns CTS.
- Matching receipt ID with wrong/missing command name and duplicate root/terminal fields fails pending and kills child BEFORE disposal.
- Native-to-managed parity after RoPE denominator reuse. cpuPreparation benchmark both orders, no fixed speed threshold.
- Dataset2690 and ten old holdouts+48 basics byte-identical; all records fit512. Staged zero/basic/conversation remains unmodified.
