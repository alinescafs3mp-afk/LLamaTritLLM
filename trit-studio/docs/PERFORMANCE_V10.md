# Audit10 performance validation

This is a methodology and executable benchmark source, not measured native results. No .NET compiler/runtime
was available in the authoring container. Python array equivalence and shape counts are not C# throughput.

## New textPreparation benchmark

`TrainerBenchmark` now includes EncodingBenchmark.Run(reverse) in `textPreparation`. It uses128 deterministic
synthetic examples spanning plain text/single-turn/multi-turn and multibyte characters, no user workspaces.
The legacy list/byte-array/LINQ implementation remains ONLY as test/benchmark oracle. Before measuring, verify
exact input and label arrays. Warmup3 rounds; measure16 rounds in each mode. Reverse order option remains.
Measure synchronous single-thread Stopwatch time and GC.GetAllocatedBytesForCurrentThread around actual
preparation loops. The legacy oracle keeps the same static UTF8Encoding instance as v9, not a fresh encoder per fragment.
Check target/token counts as functional evidence. Pool warming is explicitly part of warmup.
No speed ratio is mandatory: tiny inputs or an implementation regression may make an optimization slower.
Report actual output and diagnose rather than changing assertions. This measures host preparation, not CUDA.

## Retained gates

Existing explicit/SDPA, uniform/bucketed and selected/full output comparisons, native synchronized CUDA
training timings, evaluation preparation cache, checkpoint JSON cache, v5/v6 packing parity/timing and
portable inference allocation probes remain unchanged. Full snapshot verification is not reduced.

## Representative real measurements

On the prepared Grok host run the normal delivery entrypoint. Preserve logs and benchmark JSON in artifacts.
On the target laptop run Check-laptop.cmd. Compare warm/cold preparation, initial creation, basic and conversation
stages, accepted online update and refused material action. Record device, dtype, runtime, model shape, corpus
version, sequence/batch settings and thread count. Repeat in both trial orders, separate IO-heavy checkpoint work.
Direct encoding reduces intermediate lists but still validates/counts Unicode and fills causal input arrays.
The128-record parallel threshold is an engineering heuristic. Test1/2/bounded threads and adjust only with evidence.
Train/eval/reopen soak is required for native lifetime stability; RSS pools are neither proof of a leak nor proof of absence.

## Failure performance

A refused preflight must not rehash/encode/upload/publish the unchanged native model. An actual mutation failure
still requires complete restore. Count real Published events and preserve earlier recovery fault tests.
A10s default bound applies separately to write-slot wait and transfer/flush. It does not apply to command completion.
Test750ms transport with2s valid completion, and a child that never reads a900000-character legal command.
Timeout means delivery is unknown, not that training safely rolled back. Do not retry automatically.
