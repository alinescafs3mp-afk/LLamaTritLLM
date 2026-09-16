# Trit Studio: performance, stage integrity and recovery audit 8

## Evidence boundary

Full cumulative source revision of the supplied audit7 package. No owner checkout or remote was changed.
Authoring environment: Python/PyTorch CPU; `dotnet`/C# compiler unavailable, official SDK request failed DNS resolution.
**C# build/tests, native TorchSharp, Avalonia, Windows and CUDA: NOT_RUN.** Independent Python specifications do not execute this application.
Grok must run the real build/runtime gates and repair genuine integration failures. Historical audit reports are retained as historical evidence.
The pinned original integration base is inherited from the verified v7 package; the read-only preflight must inspect the actual current fork.

## Findings implemented in source

| ID | Finding | Change / required real regression |
|---|---|---|
| A8-01 | A terminal response could remove its TaskCompletionSource before parsing a malformed error field; fatal cleanup could no longer resolve that request | Strict `CommandCompletion.Parse` BEFORE dictionary removal. Boolean success/cancel, optional string/null error, contradiction rejection. Two real child modes with matching request IDs and invalid terminal payloads must resolve failure and terminate before Dispose. |
| A8-02 | Every real validation pass rebuilt sorted supervised input/target/position arrays, including after unchanged dataset preparation | Owned `EvaluationBatchCache`, lazy CPU-only array reuse, 16 MiB retained payload limit. Reference identity invalidates on corpus replacement, resource/session replacement resets cache, cancellation covers hits. Actual weights are still evaluated after updates; this is NOT a loss-result cache. |
| A8-03 | Offline operations encoded the entire durable pending+learned replay ledger, even when only a few records or no replay were consumed | Remove full historical materialization/encode pass; validate actual selected inputs, bounded recent replay and online pending at consumption. A clean basic stage cannot be blocked by an unused legacy long message. Existing replay contamination guards remain. |
| A8-04 | First conversation training with bundled updates disabled could relabel a basic-only model as conversation-trained | Validate stage material before mutation. Require actual bundled conversation material or confirmed prior conversation training. Nullable durable provenance flags distinguish legacy unknown state from proof. Default bundled checkbox remains on. |
| A8-05 | Initial custom stage could similarly use only previous basic material; basic/untrained UI silently discarded selected custom paths | Require selected or previously trained custom material for a custom stage. Reject incompatible selected files with explicit UI/worker feedback, not silent ignoring. Baseline still consists solely of its 48 text records. |
| A8-06 | Reusing immutable stage copies emitted the current revision instead of the archived revision | `StagePreservation` receipt records created/reused plus ACTUAL immutable stage metadata. UI logs existing first reference separately from current working weights. Reject new zero-stage refs with nonzero steps, and trained-stage refs with zero steps. Existing first copies are not overwritten. |
| A8-07 | Rollback activated the parent pointer before semantic data decoding/native optimizer restore completed | Prospective full load and restore with no publication or replay mutation; activate only on success. Failure reloads the still-authoritative current checkpoint. Real worker fixture corrupts parent counters while updating hashes; failed rollback must preserve the original active pointer. |
| A8-08 | A historical best validation score persisted across a changed sequence budget, despite changed effective inputs | Record ValidationSequenceLength; keep historical best only for the same budget. Legacy absent field is unknown, not proof of comparability. The CURRENT before/after guard always runs. This does not alter the workspace's validation records or certify cross-workspace comparison. |
| A8-09 | Bundled manifest parsing was unbounded and cancellation did not reach hashing/loading consistently | Bounded 64 KiB manifest stream, cancellation through the loader/hash/dataset path, allow-listed v8 split. Pre-cancel must fail before file access. Ordinary imported JSON size/count limits remain. |
| A8-10 | Cache optimization had no isolated target-side allocation comparison | Actual C# benchmark adds warmup-excluded, both-order zero-cache vs retained-cache CPU array preparation; verifies complete target coverage. Native tests check weights force real re-evaluation while prepared arrays are reused. No speed ratio is mandatory. |
| A8-11 | Corpus evolution risked changing the baseline against which the owner compares growth | Freeze all 48 basic pretrain records byte-for-byte and assert their hash in deterministic generation/validation. Preserve all six previous holdout files. Add 160 seed, 24 controls and 16 new challenge records. |

## Executed evidence

- `reports/dataset-audit-v8.json`: 1514 conversation train, 184 controls, 144 holdouts, 48 frozen basics; total 1890. Schema, hashes/counts,
  UTF-8, normalized/input-history splits and sequence limits passed. Largest complete sequence remains 437 byte tokens, fitting default 512.
- `reports/audit8-reference.json`: independent Python checks of 10 receipt states, 9 cache/budget/batch combinations with two different
  weight tables and complete target coverage, 6 rollback state/fault simulations; frozen baseline data. NOT native concurrency/filesystem evidence.
- `reports/numerical-reference.json` and `reports/performance-reference-v8.json`: retained independent neural math, real Python toy gradients,
  SDPA/projection/QAT equivalence and corpus shape counts. Not C# throughput, GPU savings or conversation quality.
- `reports/static-audit-v8.json`: XML/references/source invariants/script syntax/version coherence; **172 defined managed contracts**, not passed C# tests.
- The enclosing cumulative records full/previous-patch reproduction and refusal cases on throwaway local Git repositories only.

## Required real acceptance, not executed here

All 172 managed contracts plus existing headless UI/real worker tests; now six broken-child modes including malformed terminal responses.
Zero->basic->conversation->restart, missing-material refusal and durable provenance; immutable archive receipt, failed rollback pointer invariance;
native loss/gradient/optimizer tests; prepared cache retention/hit tests with real weight updates; repeated create/train/evaluate/reopen soak.
Run CPU host gates before publishing win-x64 CPU+CUDA portable ZIP. Run true Windows desktop and RTX5080 CUDA tests on the laptop.
No canned replies, weakened assertions, invented passes, silently omitted CUDA, or stale ZIP reported as a fresh build.

## Residual limitations

The 16 MiB limit covers retained long-array payloads, not source corpus, object metadata, native allocator pools or transient allocations.
Cached arrays assume single-owner immutable input data, as did prior result caches. No GPU-resident input cache, pinned memory, mixed precision or peak VRAM claim.
Rollback temporarily disposes/replaces the session; failure recovery can itself fail and must remain visible. Atomic pointer writes are not all-filesystem power-loss certification.
A successful rollback with failed replay reconciliation emits a warning; online stays off. Queue repair is not data erasure or unlearning.
Old LastMaterial alone is not trusted provenance. First old-workspace conversation/custom resume may require explicit bundled data/selected files.
Stage names indicate actual material and optimizer steps, not learned skill. A first immutable reference is not the most recently trained model.
The frozen 48-row baseline is intentionally tiny; 1890 corpus records include controls and tests and do not prove fluent conversation.
Basic/controls/holdouts are distinct. Exact normalized guards do not catch all semantic paraphrases. Chat/replay remain plaintext.
Binary model format stays Trit Studio V1; extra backward-compatible JSON metadata does not add upstream `.bin` import.
