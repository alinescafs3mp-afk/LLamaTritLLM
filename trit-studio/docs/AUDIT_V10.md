# Trit Studio: performance and failure-path audit10

## Evidence boundary

Full cumulative source revision of supplied audit9. No owner checkout or remote was changed.
Executed here: deterministic dataset/schema/hash/partition checks, independent Python neural/encoding specifications,
structural/script checks and local synthetic Git package rehearsal. **C# compilation/tests, native TorchSharp,
Avalonia, Windows and CUDA: NOT_RUN.** No dotnet/C# compiler is installed in this authoring environment;
a direct official SDK metadata request failed DNS. Grok must compile, repair actual integration issues, and run all gates.
Historical reports are retained as historical evidence. Python parity is NOT execution of the C# implementation.

## Findings and implemented source repairs

| ID | Finding | Repair and required actual regression |
|---|---|---|
| A10-01 | Token creation allocated an intermediate byte array, LINQ iterator and int array per text fragment | EncodeInto writes final int span using small stack scratch or cleared pooled byte buffer. Strict UTF8/capacity check precedes mutation. Native managed exact legacy parity and offset/invalid-input tests. |
| A10-02 | Training/prompt preparation built growable full token lists, then copied inputs and labels | Premeasure retained whole history and complete target, allocate final arrays once, fill roles/text and shifted labels directly. No context/target/mask semantic change. Exact whole corpus and trimmed-history oracle. |
| A10-03 | Tiny online encoding pools entered parallel scheduling despite little work | Sequential path below128 examples or one worker; larger pools chunk32 with bounded parallelism. Cancellation and ordered result equivalence required. Threshold is heuristic, not a measured optimum. |
| A10-04 | An alive child that stopped reading stdin could block command write indefinitely | Bound semaphore wait and write/flush separately; cancel underlying I/O, also bound await, observe abandoned fault, fail pending receipts and kill child. Default10s per transport phase, NOT a training duration limit. No auto-retry of ambiguous commands. |
| A10-05 | Mode enabled in memory before runtime.json failed to persist; failed command could leave hidden online learning | On IO/permission failure pause/disable in-memory online work, emit authoritative mode-state and explicit storage error. Old disk flag may remain: repair before reopening; do not claim a successful durable write. |
| A10-06 | Refused train input always triggered full rehash/encode/native restore/upload of an unchanged model | Track whether session/settings/RNG/optimizer could have mutated. Skip full recovery before that boundary, preserve complete recovery after it. Actual-worker test counts publication events on rejected material; earlier mutating-failure tests retained. |
| A10-07 | Preparation optimization lacked isolated runtime evidence | Native textPreparation benchmark retains legacy list implementation only as oracle, checks all arrays outside timing, warms up, measures both orders and thread allocations. No speed ratio gate or fabricated native result. |
| A10-08 | Data evolution needed new continuity without shifting baseline | Add160 training,24 control,16 challenge records. Keep48 basics and eight earlier blind files byte-identical. Versions, loader and tests advance together. |

| A10-09 | Hand-maintained split lists in managed tests could miss a newly added challenge | Enumerate manifest entries for rejection/sequence/hash/count coverage, including the frozen baseline. Unknown runtime files still fail the allowlist. |

## Performance and failure semantics

Final arrays and causal context are still fully present. Span encoding is not zero-allocation overall; larger temporary byte
buffers come from ArrayPool and are cleared on return. The tokenizer is unchanged UTF8-byte vocabulary.
Native training math, formatV1, checkpoint verification, validation checks and optimization cadence are unchanged.
Transport waits may include waiting for a preceding write plus the actual transfer; each phase has its own10s bound.
A successfully delivered request can execute longer than that. The timeout is not an inference/training timeout.
After timeout the command may have been partly or fully received: fail the connection, never blindly replay it.
Killing a stalled child can discard unpublished work. Recover from the last fully committed verified checkpoint.
Runtime-mode write failure pauses only the live process with certainty. Persistence cannot be guaranteed on failed storage.
A disk repair/reconnect workflow must show the saved mode; this patch does not promise old flags were erased.
Nonmutating input refusal still gives visible error/paused feedback. Once settings, RNG or weights can change,
full checkpoint recovery remains necessary. No performance shortcut waives integrity or native restoration after mutation.

## Executed independent evidence

See reports/dataset-audit-v10.json, audit10-reference.json, numerical-reference.json,
performance-reference-v10.json and static-audit-v10.json; enclosing evidence/package-rehearsal-v10.json.
Direct-encoding specifications compare all2290 dataset records and4992 valid randomized/budget cases, plus21 buffer
boundaries. They are independent Python specifications, NOT the C# encoder, OS pipes or runtime concurrency.
Corpus:1834 seed,232 control,176 held-out,48 basic. All whole sequences <=437, fit512. Old data hashes are preserved.
Structural audit counts218 defined managed contracts, not passed tests. Source/package counts come from their manifests.

## Actual Grok/laptop gates, pending

Run all218 managed tests, all retained headless/worker/native gates and two new real child transport scenarios:
blocked stdin with legal large payload must fail and child must die before Dispose; delayed valid completion MUST succeed
even when it exceeds the configured transport deadline. Real worker must remain paused/off after forced runtime.json
write failure, retain queued data/weights, and avoid publication/native reload after nonmutating material refusal.
Run native textPreparation benchmark both orders, prior neural/QAT/optimizer parity, repeated train/eval/reopen soak,
then Windows GUI and actual RTX5080 CUDA tests. CPU fallback is not a CUDA pass. No softened assertions/canned answers.

## Residual limits

No C# speedup, peak RAM/VRAM reduction, leak-free soak, compiled application or pretrained model is claimed.
Partial command transfer can have ambiguous execution; durable workspace state is authoritative after reconnect.
Mode persistence failure may leave prior on-disk flags. Disabling live training does not unlearn weights or erase raw journals.
Corpus remains small/synthetic; exact partition guards are not semantic-independence proof or fluent-conversation evaluation.
Byte tokenization, FP32 training, independent CPU inference, explicit checkpoint cleanup and plaintext local journals remain.

## Implementation references consulted

Official .NET10 API contracts, accessed2026-09-16:
- https://learn.microsoft.com/en-us/dotnet/api/system.io.streamwriter.writelineasync?view=net-10.0
- https://learn.microsoft.com/en-us/dotnet/api/system.threading.tasks.task.waitasync?view=net-10.0
- https://learn.microsoft.com/en-us/dotnet/csharp/asynchronous-programming/cancel-async-tasks-after-a-period-of-time
WaitAsync bounds waiting; it does not by itself certify underlying I/O cancellation. The implementation requests cancellation
and terminates the unusable process separately. Actual platform behavior requires the real child tests.
