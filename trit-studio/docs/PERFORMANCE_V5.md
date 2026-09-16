# Performance methodology: audit5

## What changed

Only supervised final states reach the final RMSNorm and vocabulary head when `ProjectOnlyTargets=true`.
Earlier user/history/padding states still participate in transformer computation. This is mathematically valid here because
final RMSNorm and the linear vocabulary head act independently at each sequence position and ignored logits do not contribute to loss.
Tied embedding gradients still combine input-embedding and selected-head paths. No targets are discarded or reweighted.
The switch defaults on; old resource JSON defaults to on. Disabling it selects the full-output reference path.

Plane-local scopes/in-place temporary arithmetic reduce QAT temporary lifetime. The residual is a clone of detached weights:
using only detach and subtracting in place would share master storage and is prohibited. Layer scope boundaries must preserve autograd
saved tensors through native references. Actual C# forward/backward and repeated native-memory tests are required.

CPU generation shares immutable RoPE across sessions while retaining independent KV/scratch state. Logit and sampling arrays are reused
within a response. The public unbuffered Step and Sample APIs remain available for compatibility and comparisons.
Online training samples base/replay without copying all base references; loading a new session avoids a second master upload.
The split guard caches normalized validation keys, rather than hashing the whole validation corpus on each chat message.

## Reproducible authoring checks (executed here)

```
python3 tools/build_conversation_dataset.py
python3 tools/validate_dataset.py
python3 tools/numerical_reference.py
python3 tools/performance_reference.py
python3 tools/static_audit.py
```

Python uses CPU PyTorch 2.10.0 in the authoring environment. These are numerical/structural specifications, not C# benchmarks.
Reports disclose scope. The performance report contains four selected/full gradient comparisons, 18 QAT comparisons, prior attention
checks, corpus target-position counts and bucket shape simulation. It makes no FPS/tokens/s or VRAM claim for Trit Studio.

## Mandatory actual native evidence (Grok/target)

From the delivered package, Check-laptop.cmd runs tests and benchmarks, including CUDA with an actual device. CPU host delivery
executes the corresponding host gates. Running a CPU fallback does not satisfy `--cuda`.

```
TritStudio.Trainer --self-test
TritStudio.Trainer --benchmark --output cpu-benchmark.json
TritStudio.Trainer --benchmark --reverse --output cpu-benchmark-reversed.json
TritStudio.Trainer --self-test --cuda
TritStudio.Trainer --benchmark --cuda --output cuda-benchmark.json
TritStudio.Trainer --benchmark --cuda --reverse --output cuda-benchmark-reversed.json
```

Native timing excludes warmup and synchronizes CUDA around measured intervals. The three modes are:
1. explicit attention + uniform sampling + full vocabulary output;
2. SDPA + length buckets + full vocabulary output;
3. SDPA + identical bucket policy + selected vocabulary output.
Compare 2 vs 3 to isolate output projection, including the cost of position preparation/gathering. Compare 1 vs 2 only as a combined
attention/batching change. Repeat both orders on the same machine. Report dataset/shape/backend/thread settings and actual target tokens/s.
A 262-token vocabulary is small; additional gather overhead can outweigh saved matrix work for some configurations. Retain the switch.

The managed allocation probe runs independent 32-token sessions, after initialization, using one CPU thread. It reports allocated bytes
and elapsed time for unbuffered/buffered Step; it is not process RSS, peak live heap, whole-chat generation cost or device VRAM.
Repeat after JIT warmup before drawing conclusions. Do not require a fixed speedup ratio to pass functional acceptance.

## Soak and correctness requirements

- Run both optimization flags with target-loss parity on variable lengths, multiple QAT planes and GQA settings.
- Compare parameter gradients, not only outputs. Assert master unchanged by forward and changed by real optimizer steps.
- Repeat train -> evaluate -> save -> reopen -> train. Validate optimizer moments and sampler state, not only weights.
- Observe RSS after warmup across repeated cycles. Use actual native/device memory tools on the target for VRAM, not host RSS as a proxy.
- Burst UI updates while scrolling old messages. Terminal feedback must stay final. No waiters or active child survive fatal framing errors.
- Report failures with logs. Do not replace models with canned replies or weaken tests to produce an artifact.
