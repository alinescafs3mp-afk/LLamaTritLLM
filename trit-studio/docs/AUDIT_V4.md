# Trit Studio: source audit 4 · 2026-09-16

## Evidence boundary

Cumulative revision of audit3, prioritizing training/runtime overhead and consequential failure paths.
The author executed corpus checks, independent Python/PyTorch numerical/algorithm checks, source/script checks and local Git package rehearsals.
**C# compilation, the defined C# tests, native TorchSharp execution, desktop rendering, Windows execution and CUDA are NOT RUN here.**
There is no .NET SDK/compiler in this authoring environment. Grok must compile, repair genuine integration failures and pass all host gates.
A synchronized native C# benchmark is included but its output is generated only when Grok/the laptop actually runs it.
Historical AUDIT_V2/V3 files remain historical evidence, not current test results.

## Implemented findings

| ID | Finding | Repair and required regression |
|---|---|---|
| A4-01 | Every validation batch reconstructed all fake-quantized matrices | Scoped no-grad matrix cache per whole evaluation. Cached tensor owners are explicitly disposed. Native test counts one construction per quantized matrix. |
| A4-02 | The next publication cycle repeated the just-computed unchanged validation | Exact corpus-reference + model-weight-version result cache, invalidated by update/restore. A failed evaluation does not populate success cache or poison ordering. |
| A4-03 | Each validation batch forced a device scalar read | Accumulate token-weighted losses on-device and read one scalar at the end. Loss and gradient norm share one host read per training step. |
| A4-04 | Uniform-length mixing spends compute on right-padding | Example-weighted length buckets, preserving per-example marginal selection instead of selecting small buckets disproportionately. Forced online corrections retain mixed replay; option can be disabled. |
| A4-05 | Explicit attention materialized scores/masks in the managed training expression | Optional/default-on LibTorch SDPA, causal and dropout=0, with existing explicit reference path. GQA expansion retained. No FlashAttention or mixed-precision guarantee. Native forward/backward parity gates added. |
| A4-06 | Full master copied GPU->CPU before every candidate, even when last committed master already exists | Retain immutable committed master baseline in worker; candidate copied once at publication. Preserve full optimizer/corpus verification for restore. |
| A4-07 | Chat snapshot activation rehashed master/optimizer/data and packed file repeatedly | Hash packed file once and parse the same held stream. Verify metadata. Trainer/rollback keep full snapshot verification. Suppress duplicate publication loads. |
| A4-08 | Streaming callbacks could queue old text behind the finished answer | Latest-value mailbox bounds dispatcher work and invalidates queued partial text before terminal display. No per-token backlog on slow UI. |
| A4-09 | Malformed trainer stdout only logged, leaving pending commands unresolved; EOF waited for process exit | Invalid JSON/envelope/completion without an ID and unexpected EOF terminate the client protocol and fail pending requests immediately. Real child fixtures test malformed JSON, missing fields and EOF while child stays alive. |
| A4-10 | Every status and online batch enumerated/materialized the full replay ledger | Cached replay state counts and bounded pending/recent slices; no-op reconciliation avoids rewriting unchanged state. Mutating persistence remains an atomic full JSON snapshot, not an O(1) log. |
| A4-11 | Tiny-step status floods competed with UI and pipe reads | Coalesce same-stage busy progress to at most 4 Hz. Stage/terminal receipts are not dropped. UI shows effective batch, padding, backend and validation-cache diagnostics. |
| A4-12 | CPU attention divided by the same denominator for each channel; tiny quantizers entered Parallel.For unnecessarily | Normalize scores once per head, traverse values contiguously; sequential packing for small tensors. Parallel determinism and numerical fixture tests retained/expanded. |
| A4-13 | UTF-8 length preflight allocated token arrays; prompt assembly repeatedly inserted at the list front | Strict allocation-free UTF-8 count, build retained history once in chronological order, do not encode discarded pairs. JSON datasets/state parse from streams. |
| A4-14 | Export could replace live workspace metadata or follow a linked destination | Refuse same source and any workspace descendant/linked destination; sibling exports remain valid. Tests cover source, state files and links. |
| A4-15 | Picking/clearing datasets could race a model switch; delayed teaching receipt could target changed UI state | Dataset picker participates in navigation gate; clear disabled while busy. Teaching receipt is checked against captured client/workspace generation. |
| A4-16 | Some malformed numerical/state inputs failed late or could be persisted unreadably | Reject nonfinite sampling logits, invalid unpack shapes, quantizer scale overflow, forged replay IDs/states, invalid revision metadata. Symmetric JSON read/write budget preserves old valid state on overflow. |
| A4-17 | Manual fine-tuning could omit recently learned corrections | Include up to 256 recently retained learned examples in manual corpus merge, excluding validation IDs. This is replay mitigation, not a no-forgetting guarantee. |
| A4-18 | No repeatable target-side performance evidence | Isolated --benchmark with warmup, synchronized CUDA timing, optional reversed trial order, padding/loss/RSS reports. Host delivery and laptop diagnostics run it. No hard timing ratio is an acceptance gate. |
| A4-19 | Starter conversation coverage needed another substantive increment | 746 train, 88 control, frozen old 40 test + 24 challenge, new 16 challenge = 914. All full sequences <=279 byte tokens. Old blind files unchanged byte-for-byte. |

## Executed independent evidence

- `reports/dataset-audit-v4.json`: schema, UTF-8, manifest/hash/count, exact/normalized disjointness and full input-history checks; 914 records.
- `reports/numerical-reference.json`: analogous PyTorch dense vs NumPy incremental GQA, maximum absolute logit error 2.0862e-7; toy gradients really update Python weights. This is NOT execution of C#.
- `reports/performance-reference-v4.json`: explicit vs SDPA forward/backward on four independent Python shapes (including GQA). Largest forward difference 4.7684e-7; gradient difference <=9.3133e-9.
- Python shape simulation on bundled training lengths: uniform batches pad 26.87% of positions, buckets 6.66%; 21.59% fewer total padded token slots and 35.52% fewer dense attention shape slots. **Not measured C# throughput or a promised speedup.**
- `reports/static-audit-v4.json` and outer `evidence/package-rehearsal-v4.json`: scopes are explicitly structural/packaging, not compiler/native checks.

## Defined acceptance, awaiting Grok/target execution

64 managed contracts plus headless UI/three actual broken-child protocol cases, existing real worker lifecycle checks,
expanded real trainer SDPA/explicit forward+gradient/cache/failure/resume checks, and a real native benchmark.
The CPU host delivery must complete the whole suite before creating an accepted Windows portable ZIP.
The laptop must run CPU/CUDA self-tests and benchmarks, then manual desktop checks. CPU fallback cannot be called a CUDA pass.

## Residual limits

- No actual C# speedup, compiler success, native leak-free soak, GUI pass or GPU pass is claimed here.
- FP32 weights/activations and expanded GQA remain. SDPA dispatch depends on device, dtype and installed native libraries; explicit mode is available.
- Validation cache assumes single-owner immutable encoded corpus arrays and weight updates through the training session. Replacing data requires a new encoded array.
- Benchmark timing is synchronized; ordinary per-step UI milliseconds are host-side instrumentation, not an isolated GPU kernel timer.
- Repeated validation is a candidate guard, not a blind measure of generalization. Buckets preserve marginal sampling but change batch correlation.
- Cached master and eval weights trade bounded RAM/VRAM for repeated work. Native allocators can retain pools; host RSS is not VRAM telemetry.
- Replay writes remain whole-document snapshots; its 50000-record/128 MiB bound is not a scalable database. No automatic live checkpoint pruning was added.
- Raw chat/replay remain plaintext; clearing a queue is not erasing history or unlearning weights. The exclusion control is not secret redaction.
- Cancellation takes effect between native operations; corrupted/overlong line checks are not hard real-time OS resource isolation.
- JSON write budget is checked before commit, but a temporary file may grow to serialized payload size before rejection.
- Packed-only chat loading intentionally does not certify optimizer recoverability; full training restore still checks every required file.
- No pretrained weights, original `.bin` importer, distributed training or general conversational fluency claim. Dataset tests do not establish semantic independence.
- Source binary format remains Trit Studio v1; this does not make it compatible with the upstream console model format.
