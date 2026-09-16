# Trit Studio: performance and failure-path audit 9

## Evidence boundary

Full cumulative source revision of the supplied audit8 archive. No owner checkout or remote was changed.
Executed here: Python corpus and algorithm checks, independent PyTorch/NumPy neural specifications,
source/script checks and synthetic local Git package rehearsal. **NOT RUN: C# compilation/tests,
TorchSharp runtime, Avalonia, Windows or CUDA.** No .NET SDK/compiler is available in the authoring
container; an official SDK metadata request failed DNS resolution. Grok must execute every real gate.
A passing independent Python specification does not certify these C# implementations or native memory lifetimes.
Previous audit reports are retained as historical evidence, not results from this version.

## Implemented changes and required acceptance

| ID | Finding | Repair / actual regression required |
|---|---|---|
| A9-01 | Each online/offline snapshot serialized and reread identical training/control JSON again | One-slot `CheckpointJsonCache` per corpus; exact owner-reference identity, hash during first streamed write, then cached exact bytes+digest. At most4MiB retained payload per slot,8MiB total. Every snapshot still has independent complete files. Changed array invalidates. Native C# byte/hash/cache tests and disk benchmark required. |
| A9-02 | An oversized cache payload could encourage a costly second serialization or unbounded memory | Bounded recording tee relinquishes its buffer on overflow but continues the SAME serialization. Zero-budget streaming mode. Cache only publishes after successful write/flush; cancellation or serialization failure removes only a newly-created partial file. Existing destinations are never replaced. |
| A9-03 | Valid hashes did not prove metadata and state belonged to the same training step | Before native replacement/encoding, require state Step/TargetTokens to agree with revision, positive sampler, feasible changed-weight count and valid unique committed IDs. Zero-stage traces are refused. Applies on normal restore, prospective rollback and resource reconfiguration. Hash-valid positive-step mismatch fixture must retain previous active pointer. |
| A9-04 | TXT/JSONL used replacement-fallback decoding and read an entire arbitrarily long line | Strict UTF8 line reader, optional UTF8 BOM, no silent UTF16/32 reinterpretation, bounded accumulation before parse. TXT32768 UTF16 chars per line, JSONL1Mi chars. Decoder errors identify file/approximate line. Cancellation is not wrapped as malformed data. Ordinary JSON array parsing remains System.Text.Json. |
| A9-05 | Batch selection rescanned every selected sequence's labels on every step | Per-corpus length+target-count metadata. Choices and RNG calls unchanged, including uneven buckets, forced online corrections, ignored tails and duplicate draws. Required outside-corpus example is scanned separately. Preparation cancellation included. |
| A9-06 | Changed-weight telemetry scanned the full model on one thread and could not be stopped | Bounded tensor-parallel exact comparison for larger models; sequential small-model path, preflight equal config, per-chunk cancellation, shared-array fast path. Exact results and cancellation tests; no approximate changed count. |
| A9-07 | A successful answer followed by a disk append failure was reported as failed generation, restoring the already-answered prompt | Completed response and in-memory turn remain. Separate visible journal warning, no restored prompt, no fake durable-save claim. Headless fixture creates a directory at chat.jsonl to force a real append failure after actual CPU generation. Not an automatic disk retry or crash-persistent queue. |
| A9-08 | Existing dataset audit could miss equivalent input forms across JSON record schemas | Normalize runtime input shape for single-turn and messages forms when checking train/control/test partitions. All v9 data passes; no semantic-equivalence or generalization claim. |
| A9-09 | Corpus growth must not alter the owner's pretrain comparison point | +160 train,+24 controls,+16 new held-out. All7 prior holdouts and48-row baseline byte-frozen. Corpus/app/package versions advance together. |

## Performance semantics

Cache ownership is deliberately narrow: the worker owns immutable `TrainingExample[]` arrays; replacing
an array invalidates its cached serialization. Mutating an array or nested history in place is unsupported,
as it already was for prepared/evaluation caches. Files are fully written and flushed on EACH snapshot;
this optimization is not deduplication, hard-linking, less fsync, removal of checkpoint verification, or pruning.
The cap covers retained byte arrays only. Capture buffers, final-array conversion, serializer buffers,
source corpora and allocator metadata can temporarily coexist. No peak RSS/VRAM number is claimed.
Hashes cover intended written bytes; subsequent full recovery still rehashes them. No signature or
all-filesystem power-loss durability certification is added.

## Executed independent evidence

- `reports/dataset-audit-v9.json`:1674 train,208 control,160 holdout,48 baseline =2090 records.
  Whole sequences <=437 byte tokens, fit512 without truncation. Prior holdout+baseline hashes unchanged.
- `reports/audit9-reference.json`:12 bounded-capture simulations,4 valid+6 invalid UTF8 cases,
  1200 sampling/statistics equivalence trials,15 chunk-count cases+signed-zero check,10 state-policy cases.
  These are Python specifications, NOT C# filesystem/concurrency evidence.
- `reports/numerical-reference.json`: independent dense PyTorch/incremental NumPy logits and real toy
  gradient updates, not conversational fitting or execution of this application.
- `reports/performance-reference-v9.json`: retained neural parity/operation-shape specification on current data.
- `reports/static-audit-v9.json`: project/source/script/version checks;202 managed test cases DEFINED.
- Enclosing `evidence/package-rehearsal-v9.json`: exact full/upgrade reproduction and refusal cases in
  temporary local repositories, not a remote checkout or Windows execution.

## Real gates for Grok and laptop, NOT_RUN here

Run all202 managed contracts, headless UI including completed-answer/failed-journal case,
actual worker lifecycle/evolution/positive-step failed rollback, native gradient/optimizer/cache tests,
and synchronized benchmarks. New `checkpointCorpus` benchmark compares zero vs bounded retention in both orders,
excludes warmup and checks every output hash outside timing. Cache hit counts do not certify speedup.
Repeat create/train/evaluate/reopen under real native memory measurement. Capture failures, fix causes,
rerun without weakening assertions. CPU fallback never counts as a CUDA pass.
Windows package must still include CPU+CUDA trainers even when built on a CPU-only Linux machine.

## Deliberate limits

No pretrained weights, demonstrated general conversational quality, C# speedup, peak memory savings,
leak-free soak, compiled binary or working GPU is claimed. Strict import now requires UTF8: previously
accepted UTF16 text must be explicitly converted. Import file-size prechecks are not OS-level sandboxing.
A valid digest with counters matching metadata still does not cryptographically authenticate a checkpoint,
nor independently inspect all opaque optimizer internals. Workspaces remain trusted local inputs.
Journal-write failures leave the answer only in window memory; closing loses that unsaved exchange.
Conversation/replay remain plaintext; clearing pending work is not erasure or unlearning.
The binary model format remains Trit Studio V1, not upstream LLamaTritLLM .bin.
