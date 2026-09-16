# Trit Studio: performance and correctness audit 6 · 2026-09-16

## Evidence boundary

Full cumulative source revision of the supplied audit5 archive. No remote repository was mutated.
The original pinned fork base is carried forward; Grok must run the read-only preflight against the actual checkout.
Executed here: Python corpus checks, independent numerical/algorithm tests, structural/script checks and local synthetic Git package rehearsals.
**NOT RUN: C# compilation, the defined C# tests, native TorchSharp, Avalonia rendering, Windows execution or CUDA.**
There is no .NET SDK/compiler in this authoring container. No measured C# speedup, VRAM reduction or leak-free soak is claimed.
Source for Grok to build/test is not a compiled executable or a pretrained model. Previous audits remain historical evidence.

## Implemented findings

| ID | Finding | Implemented change / required runtime regression |
|---|---|---|
| A6-01 | Packing cloned an entire float matrix and dispatched a parallel loop for each plane | Finish all planes within a group-local 128-float stack buffer; chunk 64 groups per task. Preserve accumulation order, plane/group byte layout and thresholds. Exact legacy pack oracle + parallel equivalence tests. |
| A6-02 | Packed model activation decoded every plane sequentially | Range-parallel unpack with disjoint output groups and deterministic plane accumulation; same canonical-trit, scale and finite-result checks. Small tensors remain sequential; cancellation checked per group. |
| A6-03 | Snapshot creation wrote master/packed files then read both fully again for hashes | `HashingWriteStream` computes SHA256 over the exact bytes written, in bounded chunks. `WriteHashed` returns the digest; full restore verification is unchanged. Preserve existing destinations, remove only newly-created partial files. |
| A6-04 | Stop/close did not reach expensive packing/model reads until too late | Propagate cancellation through master copying, packing/unpacking, chunked model read/write/hash and snapshot preparation. Final cancellation point is before directory activation; once activation starts, finish commit without a false cancelled receipt. Native optimizer serialization/fsync remains uninterruptible within the call. |
| A6-05 | Init/copy/restore scopes retained transient native tensor aliases for an entire model | Tensor-local dispose scopes bound temporary staging lifetimes. Stored parameters/managed master arrays remain owned. Run real repeated train/eval/copy/reopen soak; allocator pools are not leaks by themselves. |
| A6-06 | Gradient norm held full squared-gradient temporaries until the entire step ended | Scalar in-place accumulation with a per-gradient dispose scope; keep one combined metrics transfer. Norm/reference and still-live-gradient checks on CPU/CUDA are required. |
| A6-07 | QAT computed absolute residual twice and the unused last residual | Reuse magnitude for scale and threshold; skip final residual subtraction. Master-storage ownership and identity STE unchanged. Native math/immutability regressions retained. |
| A6-08 | Response limit still performed an unused next forward pass | Defer the next step until continuation is actually needed. Record forward-step count for tests; EOS/context/UTF-8 constraints remain. Greedy sampler uses a stable max scan instead of sorting; stochastic order/RNG path unchanged. |
| A6-09 | Every generated token assembled layer key strings and looked up arrays | Prebind read-only layer/embedding/norm array references per immutable snapshot. KV/scratch buffers remain response-local. |
| A6-10 | Dataset encoding materialized old history that was subsequently discarded | Shared allocation-light retained-history planning; encode only retained pairs in chronological order. Full answer still preserved, no silent target truncation. |
| A6-11 | Full-history split hashes missed equivalent inputs after context trimming | Guard both full normalized input/history and actual retained input/history for the configured sequence budget. Rebuild when budget changes; preserve persistent example IDs. This does not detect semantic paraphrases or unlearn old weights. |
| A6-12 | Old pending replay could bypass the newer submit-time split check | Recheck durable pending and sampled learned entries immediately before encoding/training. On contamination stop online work and retain last committed weights; require explicit queue repair/discard, not silent acceptance. Actual worker restart fixture included. |
| A6-13 | Generation errors left the answer bubble saying it was generating; cancellation could look completed | Terminal bubble text/details, mailbox invalidation, propagate cancelled/error to common action feedback. Restore old input only when the new draft is empty. Incomplete answer is not added as a teaching target. Headless fault/cancel assertions included. |
| A6-14 | A managed test still expected 746 seed rows; app/corpus/assembly versions drifted | Corpus version contract shared with trainer; loader rejects mixed-version bundles; tests read manifest counts and enumerate v6 holdout. Assembly/file/package versions all identify v6, binary model format stays V1. |
| A6-15 | Conversation coverage and longer context remained narrow | Add 224 training, 24 controls, 16 new blind challenges. All four older holdout files remain byte-identical. Corpus totals 1442; every full example <=437 byte tokens, fits default512. |
| A6-16 | No isolated measurement for packing/staging changes | Native benchmark retains the actual v5 packer solely as a reference, compares exact bytes, warmup, 1/bounded parallel threads, timing and all-thread managed allocation traffic. No speed ratio is a pass condition. |

## Executed evidence in this environment

- `reports/dataset-audit-v6.json`: 1194 train, 136 control, 40/24/16/16 frozen blind records plus16 new; 1442 total.
  Strict UTF-8/schema/identities/hashes, normalized duplicates and input/history partition checks passed. Max full sequence437.
- `reports/performance-reference-v6-additional.json`: 91 independent pack comparisons, including a 65664-weight case,
  exact float32 scales and packed bytes; 18 QAT value/identity-gradient comparisons; 3 gradient-norm comparisons;
  36 response-limit loop simulations. They do not execute C# or parallel scheduling.
- `reports/numerical-reference.json`: independent dense Torch vs incremental NumPy max logit difference2.0861626e-7;
  analogous toy fitting loss5.62136 ->4.26310. This is not conversational evaluation or C# execution.
- `reports/performance-reference-v6.json`: retained SDPA/explicit and selected/full projection comparisons on the expanded corpus.
- `reports/static-audit-v6.json`: structure, referenced paths, source invariants, script syntax, version agreement; NOT a compiler.
- Enclosing `evidence/package-rehearsal-v6.json`: exact full/upgrade reproduction and refusal fixtures in temporary local repositories.

## Required Grok/target acceptance, not run here

121 managed core/data/persistence tests, headless UI fault/cancel receipts, real broken-child transport tests,
real worker online/restart/legacy-contaminated-replay test, native gradients/QAT/SDPA/projection/optimizer-resume tests,
hashed checkpoint verification and cancellation, synchronized benchmark including old/new packing.
Run all CPU host gates before emitting a Windows portable ZIP. Run Windows desktop/CPU/CUDA gates on the target laptop.
A CUDA fallback is not a GPU pass. Do not waive failures or replace the network with canned responses.

## Residual limitations

- Packed-file bytes should remain V1-compatible; native parity/parallel determinism needs the C# gates. It is not the upstream `.bin` format.
- Removing a full `4*N` residual clone does not remove output arrays, managed masters, optimizer states or device allocations.
  Stack buffer is512 bytes per active pack task, not the whole process memory. No peak-RSS/VRAM claim is made.
- Cancellation is cooperative between chunks/tensors/groups, not inside arbitrary native kernels, filesystem writes or optimizer saves.
- Hash-on-write certifies intended byte-stream integrity; restore still rehashes files. Hashes are not signatures or power-loss certification.
- Filesystem activation can still fail after a complete revision directory is staged; last active pointer remains authoritative.
- Full/effective normalized input isolation does not detect all semantic leakage. Old learned information is not erased.
- Chat/replay are plaintext; queue discard is not deletion of history or unlearning. Failed-generation input may already be queued as raw user text.
- FP32, expanded GQA, a fixed byte tokenizer, explicit snapshot cleanup and small starter dataset remain.
- No pretrained weights, original-format importer, external model downloads or measured general conversational fluency.
