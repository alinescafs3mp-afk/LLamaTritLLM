# Trit Studio: performance and correctness audit 5 · 2026-09-16

## Scope and evidence boundary

Cumulative source revision of audit4, based on the supplied complete archive. Prior audits are retained as historical records.
Executed here: Python corpus checks, independent PyTorch/NumPy mathematics, source/script checks and local Git package rehearsals.
**NOT RUN: C# compilation or defined C# tests, native TorchSharp, Avalonia rendering, Windows and CUDA.**
The authoring container has no .NET compiler. Its SDK download was unavailable. Grok must execute all native/managed/GUI gates;
a Python implementation is not a substitute. This package contains source, not a built Windows executable or pretrained weights.
The pinned upstream integration base is retained; preflight must inspect the actual fork before applying any patch.

## Changes implemented in source

| ID | Finding | Repair and required regression |
|---|---|---|
| A5-01 | Final RMSNorm, vocabulary projection and logits were computed for every padded/history position even though the loss masks most of them | `SupervisedBatch` prepares real targets and flat positions; select final hidden states before final norm/head. Full causal attention is retained. `ProjectOnlyTargets` defaults on; full-output reference remains available. Native output/loss/all-gradient parity is mandatory. |
| A5-02 | Invalid IDs/shapes could reach native indexing, including CUDA | Validate input/target ranges, shape, nonempty targets and target-preserving length before device transfer. No silent truncation. Test labels -100, invalid negatives, out-of-vocab IDs, cancellation and trailing ignored padding. |
| A5-03 | Temporary fake-quantization planes remained alive until the outer operation ended | Owned residual clone, in-place residual/accumulator, plane-local dispose scopes and layer-local forward scopes. Identity STE preserved. Never subtract in place from a detached view sharing master storage. Native gradient and master-immutability tests required. |
| A5-04 | Reopening/reconfiguring uploaded the master once in the constructor and again through Restore | Split optimizer/counter/sampler restoration into `RestoreState`. New sessions upload weights once; rollback of an existing session still restores weights. Validate state counters before native restore. |
| A5-05 | Every four-step online update copied a potentially 50000-entry replay reference array | Select from base/learned arrays as one logical range without concatenating the full base corpus. Preserve selection order and mixed new/replay semantics. |
| A5-06 | Per-response RoPE construction and per-token logits/sampling arrays created avoidable CPU allocations | Lazy shared immutable RoPE table per active snapshot, response-local reusable logits and sampler buffers. Stable tie order matches the v4 reference. KV state stays isolated per conversation. |
| A5-07 | Live context preview built a complete tokenized prompt on every input edit | `ByteTokenizer.Measure` computes the same retained/dropped pair counts and budget without allocating token arrays. Existing `Plan` still constructs exact tokens at send time. No claim of zero managed allocations. |
| A5-08 | Sampling could begin a multi-byte UTF-8 character with insufficient response/context capacity | Filter candidate starts/continuations against the remaining byte budget. Dynamic limit reduction can stop with an explicit UTF-8 boundary reason. Never display an incomplete scalar or claim to emit the trimmed bytes as readable text. |
| A5-09 | Protocol length was checked only after ReadLineAsync allocated the complete unbounded line | Fixed-chunk `BoundedLineReader` for stdout and command input. Detect overflow while reading. Stderr is capped and drained to preserve the next line without deadlocking the pipe. CRLF, split surrogate pairs, EOF and cancellation tests included. |
| A5-10 | Protocol failure released pending buttons but left a still-running trainer capable of modifying weights invisibly | First fatal client transport failure requests process-tree termination. Requests waiting for the write semaphore re-check failure state. Real malformed/envelope/EOF-alive/oversize child tests verify failure and exit before Dispose. |
| A5-11 | Split exclusion used whole-example IDs including the answer, allowing the same control input with a changed answer into training | Cached normalized input+history guard excludes the final answer. Apply on create, manual import, online queue and snapshot restore. Preserve legacy persistent example IDs. Reject contaminated snapshots explicitly instead of silently changing their corpus. |
| A5-12 | JSON state size was checked after a temporary file could already exceed the budget | Non-owning bounded write stream refuses each write before it crosses the staging limit. Existing valid destination stays untouched on failure; temporary file cleanup remains. |
| A5-13 | Correctness/performance comparisons did not isolate output projection or portable-buffer allocations | Native benchmark now compares explicit/uniform/full, SDPA/buckets/full, SDPA/buckets/selected. Adds separate managed CPU allocation measurement. No mandatory timing speedup threshold. |
| A5-14 | Conversation coverage needed more continuity and updated decisions | 224 new train records, 24 new controls, 16 new blind challenges. Four-exchange conversations added. All previous blind files stay byte-identical. Manifest, loader, test coverage and UI corpus version advance together. |

| A5-15 | Worker stdout/stderr continuations could capture the UI synchronization context and parse/log on the desktop thread | Start independent I/O tasks; explicitly avoid context capture. UI subscribers marshal only rendering/receipts onto the dispatcher. |
| A5-16 | CUDA discovery used unbounded ReadToEndAsync for native startup output | Bounded prefix capture with continued pipe draining, cancellation and reader-task observation; truncated stdout cannot certify CUDA readiness. |

## Executed independent numerical evidence

See `reports/numerical-reference.json`, `reports/performance-reference-v5.json`, `reports/dataset-audit-v5.json` and
`reports/static-audit-v5.json`. Scope is embedded in each report.

- Independent dense Python Torch vs incremental NumPy: max logit difference 1.6391277e-7 in this run.
- Toy fitting in Python changed 6544 parameter values; loss 5.62136 -> 4.26310. This is not a result from C# or the conversation corpus.
- Four explicit/SDPA numerical attention shapes, retaining audit4 checks.
- 18 old/in-place QAT comparisons across 1..3 planes, groups 8/32 and thresholds 0/.5/1: exact value/identity-gradient match, master unchanged.
- Four selected/full output comparisons across attention paths, lengths, GQA groups and planes: output/loss max difference 0 in these Python cases;
  largest gradient difference 7.4505806e-9. Native/CUDA rounding and lifetime behavior still require the C# tests.
- Unpadded corpus traversal: final-output positions drop from 162664 to 91803 (43.56%). Attention and FFN still process context.
  This is operation-shape counting, not total training speedup or measured peak RAM/VRAM savings.
- 1178 corpus records validated; old 40/24/16-record holdouts retain exact SHA256. Whole sequences <=279 UTF-8 byte tokens.

## Grok acceptance, defined but not executed here

94 managed core/data/persistence/framing/sampler contracts; headless UI tests including the new optimization default;
four real broken-child cases; real worker lifecycle tests including changed-answer validation leakage; native optimizer/resume,
SDPA/explicit and selected/full logits/loss/gradient checks; synchronized benchmark including managed allocation probe.

Run the complete CPU host delivery before publishing. Then run real Windows UI/CPU/CUDA checks on the RTX 5080 target.
Native tensor disposal changes require a repeated train/eval/reopen soak. An allocator retaining pools is not proof of a leak;
a low initial RSS is not proof of leak-freedom. Capture actual evidence, do not waive failing assertions.

## Residual limits / deliberate non-claims

- No C# wall-clock acceleration, peak VRAM saving, native leak-free soak, successful compile or working Windows/CUDA binary is claimed.
- FP32, expanded GQA and byte tokenizer remain. Target selection does not skip attention, create new parameters or make a tiny model fluent.
- A changed target with the same normalized full input/history is excluded; semantic paraphrase leakage and equivalence after context trimming
  are not generally detected. Loading an old contaminated workspace fails explicitly and requires reviewed repair of its training data.
- Historical replay imported by older versions may need manual review. This guard does not erase previously learned information.
- The same immutable model/encoded corpus ownership assumptions still apply to caches. Updating parameter arrays outside the owning session is unsupported.
- Fatal pipe corruption kills the child; an in-flight unpublished candidate may be lost. Recovery uses the last committed verified snapshot.
- Line limits bound accumulated UTF-16 text; StreamReader retains a small internal byte buffer. This is not process sandboxing or a hard real-time guarantee.
- Bounded JSON staging still incurs serializer/buffer memory and whole-document writes. It is not a database or a power-loss guarantee on every filesystem.
- No continuous per-token GPU autoregressive chat is added. CPU chat remains independent of the training runtime.
- Raw chat/replay are local plaintext. Queue clearing is not erasure/unlearning. No private chats or downloaded pretrained weights are in this package.
