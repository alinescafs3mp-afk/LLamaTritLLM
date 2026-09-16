# Audit16 performance verification

## Boundary
No C# or CUDA run in the authoring environment. Independent NumPy equality is not a wall-clock result.
Keep all previous benchmark sections; prefixPreparation is an additional CPU-only measurement even in a CUDA trainer run.

## Prefix last-block output elision
Construct one immutable quantized model outside timing. Compare NewSession(optimizePrefix:false) with default sessions.
Both paths create the same response-local cache/scratch buffers and suppress vocabulary logits at non-final prefix positions.
Only the new path omits unused final-block attention output/out projection/norm2/gate/up/down on those positions.
Final-block K/V and every earlier layer are identical in purpose. Do not compare against an artificially expensive baseline
that computes vocabulary logits for every prefix token: v15 already skipped those logits.

Current synthetic shapes: dimension64, FFN128, heads4/KV2, layers1/3, prefix16/128, context256, one CPU thread.
Compare logits before timing;2 warmups,4 iterations each, both requested orders. Record elapsed/allocation and actual skip count.
The benchmark includes session allocation in both modes. It is prompt prefill, not full chat latency or GPU training.
No fixed speedup ratio gate. C# numeric tests and trained native-export parity must pass BEFORE interpreting performance.

## Other changes
Metadata budgets reduce unnecessary reading on malformed input, not a normal-case benchmark claim. Negative CUDA probes
are no longer cached forever; this can add discovery work on future explicit reconnects. Successful probes remain cached.
No additional vendor runtime, threads, joined weight matrices, automatic checkpoint pruning or precision change.

Grok records raw logs/results through existing scripts/deliver.*. Check-laptop runs real CPU/CUDA benchmarks after verifying
the update and the retained runtime. Never turn a skipped test or CPU fallback into a GPU pass.
