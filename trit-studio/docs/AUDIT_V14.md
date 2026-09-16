# Trit Studio: safe workspace opening, initial publication and CPU operations audit14

## Evidence boundary

Full cumulative revision of the supplied audit13 archive. No owner checkout or remote was changed.
Executed here: deterministic data/schema/hash/split checks, independent Python/NumPy/PyTorch numerical and policy specifications,
source/script checks and synthetic local Git package rehearsal. **C# compilation/tests, native TorchSharp, headless/desktop
Avalonia, Windows and CUDA: NOT_RUN.** No .NET compiler is installed. The official SDK metadata endpoint failed DNS resolution.
Grok must execute all retained real gates. A Python policy simulation does not validate .NET scheduling, native lifetimes or GUI.
Earlier audit reports remain historical. The original integration base is inherited, not a fresh claim about the fork HEAD.
The packed binary model format remains Trit Studio V1. Source is not a built executable or pretrained weights.

## Findings and source changes

| ID | Finding | Repair and actual acceptance required |
|---|---|---|
| A14-01 | Opening a different workspace disposed the working client and cleared model/history before reading the new conversation state and model | Destination lease plus read-only WorkspacePreview BEFORE epoch change or old-client teardown. Check bounded chat-state, active path, packed checksum/model and bounded history. Commit a missing conversation ID before handoff. On preflight failure preserve old model/client/draft/history/epoch; release only the destination lease. Headless real-file failure fixtures required. |
| A14-02 | Preparing a destination could introduce another full hash/unpack/history read during activation | Reuse the verified preview WeightSet, revision and history after handoff. No second managed preview load. The newly started trainer still independently verifies recovery and may announce a newer committed revision. |
| A14-03 | First zero-step publication copied the original FP32 master back from the native device immediately after uploading it | CapturePublicationMaster reuses the immutable constructor master only while step=0,targetTokens=0,weightVersion=0. Drop reference BEFORE optimizer.step, before restore/state restore and on dispose. Native exact initial-copy comparison, cancellation, version/state invalidation and post-update checks required. |
| A14-04 | Managed RMSNorm channel scaling and residual addition were scalar despite independent channels | CpuElementwise portable Vector<float> with scalar tails/fallback. Preserve existing Dot reduction and multiply order; support exact whole-array aliases, shape checks before writes, cooperative cancellation. No new worker threads/buffers or GPU dependency. Reject nonfinite/overflowing RMSNorm reduction before output writes. |
| A14-05 | Pre-cancelled model reads could open a missing/denied file before observing cancellation | Check cancellation before file access in ModelFiles.Read,Hash,ReadInferenceRevision and WorkspacePreview. Retain cancellation during chunks and all existing integrity checks. |
| A14-06 | New copy/channel paths needed isolated target evidence | initializationAndElementwise C# benchmark compares real native CopyMaster vs eligible initial reference and prior scalar vs Vector channels. Verify values outside timing, warmup, both orders, CUDA synchronization, actual allocation traffic. No fixed speedup threshold. |
| A14-07 | Conversation growth must preserve the owner's before/after baseline | Add96 single-turn,48 multi-turn,16 text examples,24 controls and16 new blind challenges. All twelve previous blind files and48 basic texts stay byte-identical. All current full examples fit default512. Versions, loader, docs and manifest agree. |

## Memory and correctness semantics

Workspace preview deliberately holds the old model and destination CPU weights at the same time. This trades temporary RAM for
non-destructive validation. It is NOT a low-peak-memory optimization. After handoff begins, later shutdown/startup/OS failures
are not a transaction that resurrects the former native session. A native-trainer restore failure can leave verified CPU chat
available with an explicit training-unavailable warning. Optimizer recovery is NOT certified by the packed-only preview.
The preview is read-only except for the separately acquired lease; UI may create a missing destination directory/lease.
Saving a new conversation ID is an explicit UI step after successful preview and before old-session teardown.
A concurrent old trainer may publish while preparing the same workspace; the active pointer is never changed by preview, and
normal verified publication handling remains authoritative. Existing per-load cancellation/coalescing is retained.

Initial-master reuse assumes the existing single-owner immutable WeightSet contract. No caller may mutate its arrays in place.
All ordinary optimizer and restore paths invalidate the shortcut BEFORE possible partial native mutation. Model.MarkUpdated
also makes the version guard reject reuse. This removes a redundant transfer/allocation for the initial publication only;
it does not remove initialization, the native FP32 weights, optimizer setup, packing, disk writes, hashes or flushing.
Subsequent trained publications still capture the real native master. Basic pretraining is actual learning, not random creation.

Channel vectorization is not less mathematical work or a fixed speed ratio. .NET/JIT/hardware can change rounding and crossover
sizes. Use numeric tolerances and actual full-model native/managed parity. RMSNorm retains the previous square reduction;
it is not a new numerically stable sum-of-squares algorithm. Cancellation after writes begin can leave response-local scratch
partly updated; the aborted generation is not published as a successful answer. No arbitrary native operation is interruptible.

## Executed evidence

- reports/dataset-audit-v14.json:2474 train,328 controls,240 held-out,48 basics =3090 records. Strict UTF8/schema/hash/count,
  exact normalized full-input/history partition checks, frozen previous data. Maximum full sequence453 fits512, not437 as before.
- reports/audit14-reference.json:99 independent normalization cases,396 alias checks,42 residual cases,7 opening-policy states
  and9 initial-reference lifecycle states. NumPy channel outputs match the scalar specification for these inputs. Policy states
  are specifications only, not injected C# IO/native failures or scheduling evidence.
- reports/numerical-reference.json: independent dense Torch vs incremental NumPy maximum logit difference1.6391277e-7;
  actual Python toy fitting reduces loss5.62136 to4.26310. Not conversational quality and not C# execution.
- reports/performance-reference-v14.json: retained independent attention/QAT/projection math and current-corpus operation counts.
- reports/static-audit-v14.json: source/script/version checks;314 managed contracts DEFINED, not passed C# tests.
- Enclosing evidence/package-rehearsal-v14.json: actual full/previous-delta reproduction and refusals in temporary synthetic
  Git repositories. No user repository modifications. Final wrapper and nested source hashes verified after ZIP extraction.

## Required real acceptance

Run all314 managed contracts and ALL retained headless/actual-worker/native suites. New managed tests cover norm/residual parity,
whole-array aliases, tails, invalid shape/no writes, nonfinite reductions and cancellation before file access; preview bounds,
read-only behavior, malformed conversation state, wrong hash/active path and damaged journal tail.
Headless tests open two malformed destinations with an actual active CPU model: model/epoch/history/draft must survive and the
failed destination lease must be reacquirable before window disposal. Test valid workspace switching and inference-only fallback
manually as well. Run native initial-reference vs actual FP32 device-copy parity, version/state/failed-restore invalidation and
post-gradient fresh capture, optimizer resume, zero->basic->conversation evolution and exact stage archive preservation.
Execute initializationAndElementwise and every previous benchmark in both orders, plus repeated create/train/eval/reopen soak.
Host passes precede fresh Windows CPU+CUDA publication. Target Windows GUI/RTX5080 still require real laptop execution.
No weakened assertions, canned answers, CPU fallback reported as CUDA, or stale archive presented as a fresh delivery.

## Deliberate limits and references

No measured C# speedup, peak RAM/VRAM reduction, leak-free result, compiled application or fluent model is claimed.
Byte tokenizer, FP32 training, explicit snapshot cleanup, immutable first-stage inference copies and plaintext journals remain.
Exact split checks do not establish semantic independence. Related everyday subjects intentionally appear across dataset splits.
Exclusion is not erasure or unlearning. Checksums are not signatures or all-filesystem power-loss guarantees.
Official API references consulted: https://learn.microsoft.com/en-us/dotnet/standard/simd and
https://learn.microsoft.com/en-us/dotnet/api/system.numerics.vector-1?view=net-10.0 (accessed2026-09-16).
