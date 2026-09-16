# Trit Studio: performance, command lifecycle and dataset audit11

## Evidence boundary

Cumulative revision of the supplied audit10 archive. No user checkout or remote was mutated.
Executed authoring evidence is recorded in reports: deterministic data checks, independent Python numerical/algorithm
specifications, source/script structure and local synthetic Git package reproduction. **C# compilation/tests,
TorchSharp runtime, headless/desktop Avalonia, real worker lifecycle, Windows and CUDA: NOT_RUN here.**
No .NET SDK/compiler is installed in this container; direct official SDK access failed DNS. Grok must compile,
repair genuine integration issues, execute every retained gate and deliver a fresh Windows CPU+CUDA portable ZIP.
Independent Python specifications and counts of written C# tests are not runtime passes. Earlier reports are historical.

## Findings and source changes

| ID | Finding | Repair and real regression required |
|---|---|---|
| A11-01 | Every online micro-update guarded/encoded up to64 retained examples even though only up to4 could enter the mixed pool | OnlineReplay selects the EXACT legacy logical indices, then validates/encodes only selected learned rows; deduplicate repeated selected indices. Preserve pending-first ordering, base references and RNG calls. Test base/learned/wrap/long.MaxValue boundaries, selected control leakage and overlong targets. |
| A11-02 | A no-change manual training request built fresh corpus arrays and re-encoded identical training/control rows | Conservative ordered record comparison returns the prior owner only when values/metadata/history ownership are equal. Reuse encoded training only on same owner AND sequence budget; reuse controls only on same budget. Changed data/budget still rebuilds. Weight updates still trigger real evaluation. |
| A11-03 | Invalid incoming JSON/envelopes could be logged and ignored while callers waited for a completion | Strict root/envelope parsing, duplicate/case-alias/ID/type/size checks. No trustworthy correlation means cancel and close the command channel, exit nonzero; no silent retry. Well-formed unknown kinds still get correlated refusal. Actual worker malformed frames must exit before parent disposal. |
| A11-04 | A debounced online-rate edit read the current client after a delay and could affect a different workspace | Capture epoch/client/CTS; cancel on navigation/start/close, verify origin after delay. Mode application blocks incompatible model switches until acknowledged. Confirmed feedback only after successful matching receipt, no success message on a skipped edit. Retain create's explicit opt-in preference route. |
| A11-05 | Caller disposed another handler's CTS while that handler could still use it; timed-out graceful shutdown send was not explicitly observed | Timer owner disposes after finishing; cancellation helper cancels and detaches only. Keep an observer for underlying shutdown write after WaitAsync times out. Real pending-mode/headless/transport tests remain mandatory. |
| A11-06 | A long requested run could spend many updates before hitting the existing128-snapshot cap | Preflight exact worst-case ceil(steps/publishEvery), plus initial snapshot for creation, against remaining slots BEFORE encoding/device mutation. Online checks one slot before four steps. Existing publish-time cap/disk guards stay. Refuse with actionable feedback; no silent cadence change or deletion. |
| A11-07 | Preparation savings lacked an isolated comparable runtime benchmark | Actual C# onlineReplayPreparation keeps the v10 guard+encode-all64 reference, compares identical arrays outside timing, warmup, both orders, elapsed and thread allocation traffic. Count selected encodes honestly; no speed ratio gate. |
| A11-08 | Starter conversation coverage needed continued substantive growth | +160 training,+24 controls,+16 new challenge records. All nine old blind files and the48-record baseline byte-frozen. Corpus/runtime/app/package versions advance together; previous validation of existing workspaces is never silently replaced. |

## Important semantics

Selection-only replay does not weaken validation of data entering gradients. An old invalid learned record that is NOT selected
no longer blocks unrelated updates. It remains on disk and will fail if selected later; no hidden cleanup, repair or unlearning.
The base corpus and pending records are still checked at their existing boundaries. At most4 selected learned rows are encoded,
versus up to64 in v10; this is an operation count, not a claim of16x training speed. Neural steps/checkpoint cadence are unchanged.

Reuse is deliberately conservative. Equal IDs alone are insufficient. Reordered data, metadata changes or separately allocated
history require a new owner; unchanged immutable references may be reused. In-place mutation of owned corpus/history remains unsupported.
Prepared-result/evaluation/serialization caches still have the same explicit ownership and invalidation rules.

The publication check is a count preflight, NOT free-space reservation or guaranteed future IO success. Early refusal does not
modify weights or publish a new snapshot; the command's normal paused/error feedback and flags may still be persisted.
The existing128 history ceiling and explicit8-snapshot cleanup stay. Stage archives are additional files, not gradient checkpoints.

A fatal command-frame error cancels unpublished work. Existing verified snapshots remain the recovery basis; a command accepted
before a later fatal frame may already have committed. Never resend blindly or claim the previous execution was definitely absent.
Cancellation cannot interrupt arbitrary native kernels. Platform process termination requires actual runtime tests.

## Executed evidence

- reports/dataset-audit-v11.json:1994 seed,256 control,192 held-out,48 baseline =2490. Whole sequences <=437 UTF8-byte tokens,
  fit512; role/schema/hash/input/history separation and old-file hashes checked. This is not model-quality evidence.
- reports/audit11-reference.json: independent exact selection/budget/framing/UI-policy specifications with recorded case counts.
  They do not exercise .NET Task scheduling, OS pipes, CUDA or actual headless events.
- reports/numerical-reference.json and performance-reference-v11.json: retained independent neural/QAT/SDPA/projection math.
- reports/static-audit-v11.json: structural/script checks and251 managed tests DEFINED. No compiler execution.
- Enclosing evidence/package-rehearsal-v11.json: synthetic local Git application/refusal cases using actual package bytes.

## Required Grok and target gates

All251 managed contracts, all retained UI/worker/native suites, new real malformed command input tests, correlated unknown-command
refusal, no-state-change quota failure and headless stale-epoch/pending-receipt checks. The long stop-fence test still requests
100000 steps but now uses publishEvery1000 so it is a valid-capacity job rather than failing before cancellation can be tested.
Run native benchmarks including replay preparation, repeated create/train/evaluate/reopen soak, zero->basic->conversation evolution,
checkpoint/optimizer restore, partition checks and all prior regression suites. No mocked/canned model responses as acceptance.
Build both Windows CPU and CUDA trainers on the prepared host; true Windows desktop/GPU execution remains laptop acceptance.
A CPU fallback is not a CUDA pass, and a stale ZIP is not a new deliverable.

## Deliberate limits

No measured C# throughput gain, peak RAM/VRAM reduction, native leak-free result, compiled executable, pretrained weights or fluent
conversation is claimed. Unchanged corpus checks still scan record references and metadata, not zero work. Plaintext local journals,
fixed byte tokenizer, FP32 training and explicit checkpoint pruning remain. Exact split checks cannot prove semantic independence.
The48 basic texts are intentionally tiny and stay frozen. The binary model format remains Trit Studio V1, not upstream .bin.

## API contracts consulted

Official .NET10 documentation: CancellationTokenSource.Dispose must not race operations on that source;
Task.WaitAsync bounds the wait, not ownership/completion of the underlying task. The code separately cancels/observes tasks.
- https://learn.microsoft.com/en-us/dotnet/api/system.threading.cancellationtokensource?view=net-10.0
- https://learn.microsoft.com/en-us/dotnet/api/system.threading.cancellationtokensource.dispose?view=net-10.0
- https://learn.microsoft.com/en-us/dotnet/api/system.threading.tasks.task.waitasync?view=net-10.0
