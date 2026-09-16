# Trit Studio: CPU inference, publication lifecycle and dataset audit12

## Evidence boundary

Full cumulative source revision of the supplied audit11 archive. No owner checkout or remote was changed.
Executed here: Python corpus/algorithm/neural specifications, source/script structure and synthetic local Git package rehearsal.
**C# compilation/tests, actual TorchSharp, Avalonia/headless scheduling, Windows and CUDA: NOT_RUN here.**
No dotnet/C# compiler is installed; a direct official SDK download request failed DNS resolution in this environment.
Python parity is NOT execution of these C# implementations, native tensor lifetimes, real parallel scheduling or UI behavior.
Grok must run all retained build/runtime gates, fix actual integration failures and produce a fresh Windows CPU+CUDA ZIP.
Previous audit reports are historical, not current runtime passes. The binary model format remains Trit Studio V1.

## Findings and implemented repairs

| ID | Finding | Source change and mandatory real acceptance |
|---|---|---|
| A12-01 | CPU attention Q/K/V and FFN gate/up each launched separate projection scheduling; tiny MatVec calls could allocate closures despite taking the sequential branch | CpuProjection groups independent matrices into one logical row space without concatenation. One QKV group and one gate/up group; output/down remain separate. Parallel rows use16-row chunks; sequential branch keeps closures in an out-of-line parallel helper. Same per-row SIMD Dot accumulation, alias/shape checks before output writes, cooperative cancellation. Exact C# separate/fused tests and benchmark required. |
| A12-02 | RoPE initialization recomputed the identical power denominator at every position | Compute one denominator per frequency channel in managed inference and TorchModel table creation. Retain division, not reciprocal multiplication. Context/head dimensions and rotation math unchanged. Independent float32 angle equivalence plus retained real CPU/native parity gates. |
| A12-03 | Full/effective split isolation serialized and hashed identical keys twice when history was not trimmed | Reuse the full key for the same actual history. Still build two keys when trimming changes it. Preserve the exact JSON identity recipe, changed-answer exclusion, trimmed-control equivalence and old IDs. Cancellation reaches constructor, iteration and normalization; no lifetime cache of caller records. |
| A12-04 | Superseded snapshot loads completed hashing/unpacking before being discarded, delaying newest activation and shutdown/navigation | Per-load cancellation source; newer distinct epoch/revision/hash cancels waiting/reading old work. Cancellation reaches semaphore/hash/read/unpack. Same snapshot notifications share an awaited completion, avoiding duplicate loads and premature readiness. Keep last verified model until replacement. Drain old loads before releasing a workspace UI lease. Cancellation ownership remains with the async operation. |
| A12-05 | Client correlated terminal receipts by ID but did not verify the completed command kind or reject duplicate event/terminal fields | Strict EventEnvelope parsing; valid object, bounded kind/id, duplicate/case-alias rejection. CompletionCommand validates terminal fields, then exact requested kind is checked BEFORE removing pending receipt. Malformed response fails pending requests and terminates broken child under existing policy. Four additional real child faults verify wrong/missing command and duplicate identity/success. |
| A12-06 | Grouping/key optimizations lacked isolated target-side evidence | Actual C# cpuPreparation benchmark compares legacy projections with grouped variants and legacy double-key guard with reuse. Warmup excluded, exact output/decision checks outside timing, both order runs, managed allocation/timing metrics. No speed ratio is a pass requirement. |
| A12-07 | Starter conversation data needed another substantive increment without shifting evolution baseline | +160 seed,+24 controls,+16 new held-out records. All ten earlier blind files and48 basic records remain byte-identical. Builder/manifest/runtime/version/first-run docs aligned at12. Existing workspace controls remain unchanged. |

| A12-08 | A self-consistent packed snapshot was not explicitly checked against the revision/hash announced by its publication event | Compare verified revision and digest with event before activation. A hash-valid but misannounced snapshot fails without replacing the last chat model. Headless real-file regression included. |

## Performance and concurrency semantics

Grouping changes scheduling and row ownership, not the number of multiplications or network parameters. Seven layer projections
are arranged into four logical calls, not a claim of43% less math or43% faster generation. The default65536-element threshold
is a heuristic; grouping may be slower for some shapes/devices. Benchmark both, do not impose a fabricated minimum speedup.
Chunk-local output indices are disjoint. No joined matrix copy is introduced. Tiny calls stay sequential.
RoPE power evaluations drop from context*(headDimension/2) to headDimension/2; trigonometry and output table storage remain.
Split-key reuse is conditional on the exact retained-history start. Normalization/hash still occur once, not zero-allocation parsing.
The public diagnostic KeyBuilds counter is single-owner and not a thread-safe metric. Guards are not shared across worker threads.

Publication cancellation is cooperative, not instant interruption of every OS/native call. Superseded internal notifications are
not failed user actions. Duplicated notifications of one snapshot await a common outcome, including a failure. Old workspace
leases stay held until the corresponding read has drained, so maintenance cannot delete a file still read by that old UI.
This does not add a generation-time weight mutation; an ongoing answer retains its own snapshot as before.
Protocol kind checking is a consistency guard, not authentication. Trusted local child and workspace assumptions remain.
An invalid receipt may follow an already committed operation: never replay blindly; reconnect from verified active state.

## Executed independent evidence

- reports/dataset-audit-v12.json:2154 seed +280 controls +208 held-out +48 basics =2690. Full sequence maximum437 fits512.
  Hash/schema/UTF8/normalized input+history isolation checked. All ten prior blind files and baseline byte-frozen.
- reports/audit12-reference.json:24 independent projection schedule cases/4074 rows;15 RoPE configurations/194432 float32
  angles;4082 split-key pairs/decisions including902 trimmed cases;14 framing cases and5 state-machine specifications.
  These do NOT execute .NET SIMD/TPL, OS pipes, Tasks, UI rendering or real filesystem races.
- reports/numerical-reference.json: independent dense Torch vs incremental NumPy; max logit error1.6391277e-7 in this run.
  Actual Python toy updates reduce loss5.62136 ->4.26310; this is not C# or conversational training evidence.
- reports/performance-reference-v12.json: retained independent neural/QAT/SDPA/selected-target equivalence and shape counts.
- reports/static-audit-v12.json: source/project/script/version invariants;272 managed contracts DEFINED, not executed.
- Enclosing evidence/package-rehearsal-v12.json records actual package-byte reproduction/refusal on temporary synthetic Git repos.

## Real Grok/laptop acceptance, not run here

Run all272 managed contracts plus ALL retained headless, actual worker, native training and benchmark gates.
New managed tests cover grouped/uneven/parallel projections, all-or-nothing shape rejection, aliases, cancellation, old key hashes,
full/effective split exclusion, cancellation during enumeration, event lifetime/duplicates/shape and exact command identity.
Headless test holds the real load gate, supersedes a queued nonexistent revision, verifies cancellation, then duplicates a real
packed revision and requires both callers to await one activation. A misannounced digest must preserve the verified model. Run alongside existing generation/mode/navigation tests.
Run10 real malformed child output modes, all inbound malformed-command and blocked-input/deferred-completion cases.
Retain zero->basic->conversation evolution, frozen controls, no hidden online updates, failed rollback pointer invariance,
optimizer resume, selected/full logits/gradients, checkpoint hash verification and repeated native-memory train/eval/reopen soak.
CPU host success precedes Windows cross-publication; true desktop/CUDA checks occur on the RTX5080 laptop. CPU fallback is not a GPU pass.

## Deliberate limits

No measured C# throughput improvement, peak RAM/VRAM saving, leak-free soak, compiled application or pretrained weights.
Neither272 written tests nor2690 data records establish runtime correctness or general conversational fluency.
FP32, expanded GQA, byte tokenization, explicit checkpoint cleanup and local plaintext journals remain.
Split guards do not prove semantic independence. Queue clearing is not erasure/unlearning. Stage copies remain inference-only.
Corrupt storage/process failure can still interrupt unpublished work. Hashes are not signatures or universal power-loss guarantees.
