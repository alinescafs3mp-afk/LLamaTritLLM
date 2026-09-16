# Performance method · audit6

Read `AUDIT_V6.md` for the evidence boundary. The author did not run C#; all timings must be measured by Grok/the target laptop.

## Execute on real hosts

`TritStudio.Trainer --benchmark --output bench-cpu.json`
`TritStudio.Trainer --benchmark --reverse --output bench-cpu-reverse.json`
CUDA trainer: add `--cuda`. Explicit CUDA requests must fail if CUDA is unavailable; do not rename a CPU result.
The delivery tool runs the CPU host benchmark; Check-laptop runs target CPU/CUDA checks and benchmarks.

Existing training trials cover explicit/uniform/full, SDPA/buckets/full, and SDPA/buckets/selected-output.
CUDA timing is synchronized. Training warmup is excluded. Effective positions, targets, cached/uncached validation and host RSS are reported.
These are small synthetic workloads, not an estimate for every model/dataset or proof of conversational quality.

Audit6 adds `publicationPacking`: old v5 reference vs group-local packing, 262144 weights/group32/3planes, one and bounded parallel
CPU threads. Warm up each variant, then measure three calls. Exact scale/trit byte agreement is mandatory; no timing ratio is mandatory.
`managedAllocatedBytes` uses all-thread allocation traffic: it is NOT peak RAM, resident memory or VRAM and can contain runtime noise.
Unpacking time is recorded separately; the legacy unpacker is not timed as a comparative speed claim. The source baseline is test-only.

## What changed structurally

Old packing allocated a `float[N]` residual copy for each current matrix. New packing keeps512 bytes of stack scratch per active task.
The packed output scales/trits and all real model weights still exist. Parallel task granularity is64 groups, not one group per dispatch.
Init/copy/restore staging and squared-gradient temporaries are now tensor-local. Native caching allocators may retain pools.
Master and packed snapshot SHA256 are produced on write; those two complete post-write reads are removed. Optimizer/data hashes remain.
Greedy sampling avoids sorting; normal stochastic sampling is unchanged. A limit-completed answer skips one unused forward.

## Required correctness and soak before judging speed

Run every managed/native/UI/worker test first. Compare packing bytes, unpack values, logits, losses, all parameter gradients,
resumed optimizer/sampler state and snapshot checksums. Verify no param is changed by forward/QAT/export alone.
Repeat train -> evaluate -> copy/publish -> reopen using the same model, then test increasing sizes. Record warmed RSS,
process-lifetime allocation traffic, target VRAM using an external GPU profiler if available, and elapsed stage times.
Do not call allocator-pool retention a leak or stable initial RSS proof of leak-freedom. No uncontrolled GC during timed trials.
Test stop while packing/writing and close while loading a packed model. Validate the old active snapshot after interruption.
No claim of hard real-time cancellation: the native/filesystem call already in progress must return first.

## Independent reports

`performance-reference-v6-additional.json` checks Python equivalents of pack/QAT/norm/loop changes.
`performance-reference-v6.json` retains prior attention/projection checks. Neither is a C# speed benchmark.
