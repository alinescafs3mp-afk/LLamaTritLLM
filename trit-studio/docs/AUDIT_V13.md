# Trit Studio: performance, chat-file bounds and private retry audit13

## Evidence boundary

Cumulative source revision of the supplied audit12 archive. No user checkout or remote repository was changed.
Executed here: deterministic corpus checks, independent Python/NumPy/PyTorch algorithm/numerical specifications,
source/script checks and local synthetic Git package reproduction. **C# build/tests, actual TorchSharp, Avalonia,
Windows and CUDA: NOT_RUN.** No .NET SDK/compiler is installed; the official SDK metadata request failed DNS.
Grok must execute all real gates and repair genuine compile/runtime issues. Written tests and Python parity are NOT native passes.
The pinned original integration base is inherited. The package preflight inspects the actual owner checkout before applying anything.
Earlier audit/performance reports remain historical. Model format stays Trit Studio V1, not the upstream .bin format.

## Findings and implemented source repairs

| ID | Finding | Repair and required actual acceptance |
|---|---|---|
| A13-01 | The old tail reader chose a bounded start but read until moving EOF, so concurrent appends could extend the read and an oversized line allocated before checking length | Capture end once; read at most requested payload plus one predecessor boundary byte, handle short reads/EOF, propagate cancellation. Seekable-stream fixture appends on first read and must not be followed. Pool buffer is bounded and cleared on return. |
| A13-02 | Last250 history loading deserialized every record in a4MiB window even after it already had enough | Reverse LF scan and parse newest records until enough valid turns; restore chronological order. Static10k fixture parses250 instead of10000. This is a count of parsed records, NOT a40x wall-clock speed promise. |
| A13-03 | Replacement-decoded damaged bytes and malformed captured teaching history could enter loaded chat context or feedback | Strict UTF8 validation and JSON span parse, byte-size guard before decoding, nonrecursive <=8 captured turns from the same conversation with exclusion checks. Missing captured history remains legacy-compatible. Skip invalid rows with explicit inspected-tail notice, keep original file unchanged. |
| A13-04 | Error/cancel restored a previously excluded submitted prompt but left the checkbox reset, so retry could enter online training | DraftRecovery restores text plus fail-closed exclusion only when the next draft is empty. Preserve a new draft, including whitespace, and its privacy choice. Real headless failed/cancelled generation must retain exclusion. Exclusion is not erasure or unlearning. |
| A13-05 | Managed attention used scalar weighted value-channel accumulation despite independent output channels | CpuAttention uses portable Vector<float> when available, scalar tail/fallback otherwise. Preserve score dot/softmax/GQA/history and per-channel time order. No added tasks, joined arrays or GPU dependency. Validate shape/aliases before writes, cancellation per head/16 positions, reject nonfinite score sums. |
| A13-06 | New inference/tail optimizations had no isolated actual-runtime comparison | attentionAndJournal C# benchmark retains actual v12 scalar attention and forward-journal reader as reference, verifies output before timing, warms up and runs both orders. Reports SIMD width, elapsed/allocation, records parsed. No minimum speedup ratio gate. |
| A13-07 | Corpus needed more constraint tracking and calm conversation without changing stage baseline | Add160 train,24 controls,16 held-out. Freeze all eleven previous blind files and48 basics. Update loader, manifest, app/build version and active delivery docs together. |

## Memory, concurrency and privacy semantics

Journal payload cap is maxBytes(default4MiB), plus one byte to distinguish an exact record boundary. ArrayPool may retain a
larger bucket; returned ChatTurn objects, JSON buffers and metadata add memory. This is not a whole-process RSS limit.
Length capture is a finite-range read, NOT an atomic consistent snapshot under in-place overwrites/truncation. Malformed inspected
rows are skipped; no silent rewrite or recovery of missing bytes. BytesRead is payload only. SkippedRecords/RecordsInspected
refer only to inspected newest rows, not the entire historic log. Uninspected older corruption is not certified clean.
Loading touches bytes of the bounded window but stops parsing as soon as enough valid rows exist. Append still flushes to disk.

CPU SIMD targets only independent channels of weighted values. Score dot products, temporal order and network architecture stay.
Target-specific JIT lowering can differ in rounding; numeric tolerance tests apply. Vector availability is not evidence of speedup.
Cancellation is cooperative between chunks/heads, not hard real-time interruption of every OS operation.

Private retry preserves the submission's exclusion or a newly selected exclusion. A new draft is never overwritten, even whitespace.
Already trained weights are not undone. Normal non-excluded user text may be queued before generation and remains so on failure.
All raw journals/replay remain local plaintext. Excluded messages may remain in inference history, subject to existing teaching boundaries.

## Executed authoring evidence

- reports/dataset-audit-v13.json:2314 seed +304 controls +224 held-out +48 basics =2890. Max full sequence437 fits512.
  UTF8/schema/hash/count/normalized input-history splits checked. Baseline and all eleven previous holdouts byte-frozen.
- reports/audit13-reference.json:90 NumPy attention scalar/vector-chunk comparisons across GQA/tail widths,270 finite-window
  forward/reverse journal cases,16 draft/privacy truth-table cases. Static10k fixture returns/parses250 valid turns.
  These specifications do NOT execute .NET Vector/JIT, real file races, asynchronous UI or native memory management.
- reports/numerical-reference.json and performance-reference-v13.json: retained independent neural/QAT/SDPA/projection math.
  Python toy fitting is not this C# application, actual conversation training or fluency evidence.
- reports/static-audit-v13.json: structural/script/version checks;294 managed contracts DEFINED, not executed.
- Enclosing evidence/package-rehearsal-v13.json: exact full/upgrade reproduction and refusal checks in synthetic local Git repos.

## Mandatory Grok/laptop acceptance, pending

Run all294 managed cases, ALL retained actual-worker/headless/native gates and benchmarks. New journal tests cover short/growing
streams, exact/partial UTF8 boundaries, BOM/CRLF, torn/oversized/invalid rows, nested teaching-history rejection and cancellation.
Headless real CPU generation failure/cancel must preserve excluded retries while leaving new drafts and their flags alone.
Run full-model incremental/native parity and new scalar/SIMD attention comparisons on target JIT/hardware. Keep previous
publication, malformed protocol, mode persistence, cache/state restore, zero/basic/conversation and optimizer-resume gates.
Run repeated train/evaluate/reopen memory soak. Host success precedes Windows CPU+CUDA cross-publication. Actual Windows/RTX5080
acceptance occurs on the laptop. CPU fallback is not a CUDA pass. No weakened tests, canned replies or stale ZIP as fresh delivery.

## API contracts consulted

- https://learn.microsoft.com/en-us/dotnet/standard/simd
- https://learn.microsoft.com/en-us/dotnet/api/system.numerics.vector-1?view=net-10.0
- https://learn.microsoft.com/en-us/dotnet/api/system.io.stream.read?view=net-10.0
Stream.Read can return fewer bytes than requested; bounded code loops only to the captured end. Portable vector code still requires
target execution. No measured C# speedup, peak RAM/VRAM reduction, compiled binary or learned model quality is claimed here.
