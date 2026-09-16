# Performance acceptance v12: portable CPU projections and redundant key preparation

This document describes REAL C# measurements to execute, not author-side results. The source author ran only independent
Python specifications. Keep all prior benchmark sections: native training modes, packing, checkpoint cache, text preparation,
replay preparation and validation arrays. Default delivery/laptop diagnostics call the aggregate benchmark automatically.

## New cpuPreparation section

CpuPreparationBenchmark.Run compares synthetic64/256-dimensional projections with one and bounded(up to4) threads.
Legacy path runs the old MatVec per matrix with the old65536 threshold and same SIMD Dot, output arrays and data.
Grouped path runs QKV together and gate/up together; no concatenate/copy of input matrices. Both retain preallocated outputs.
First compare each float's bits outside timing. Warm up4 times, time32 repeats, repeat invocation with reversed order.
Report dimension/threads/iterations/elapsed and all-thread managed allocation delta; runtime/other-thread noise can contribute.
There is no minimum speed ratio. A negative result is valid evidence to revisit thresholds, never a reason to falsify an oracle.

The same section compares full+effective split-key preparation on32 synthetic controls and256 candidates without truncation.
Legacy computes both keys; optimized computes one. Check identical decisions outside timing, warm up, time8 passes,
report actual KeyBuilds for optimized and the known legacy operation count, elapsed and current-thread managed allocations.
The legacy arm calls the unchanged static key recipe; it does not use a deliberately slower implementation.

These cases do not establish complete-model tokens/second, UI latency, peak RSS or VRAM. Profile representative real models
and both CPUs before selecting thresholds. Do not infer43% faster inference from7-to4 logical projection groups.

## Publication responsiveness gate

Use the actual headless fixture plus a desktop run with repeated snapshot publication during a response.
Different newer snapshot must cancel obsolete pending/read work. Repeated SAME revision/hash must await the shared load,
not cancel it or announce completion early. Keep old CPU chat active until verification succeeds; failures remain visible.
Switching workspaces/cleanup must hold the previous lease until old reading drains. Record any failed assertion with logs.
A cancellation request is not a hard real-time latency guarantee; no author-side UI timings are claimed.

## Numerical gates

C# exact old/fused Dot output tests, shape/alias preflight, uneven16-row chunks,1/2/4 threads, cancellation before writes.
Retain native-to-managed logits parity including changed RoPE table preparation, SDPA/explicit and selected/full gradients.
Old/full/effective split-key hashes must match at several context budgets, including changed answer/trimmed controls.
Four new real malformed receipt subprocesses must fail and terminate before Dispose, never merely time out.

## Evidence delivered with source

reports/audit12-reference.json contains Python schedule/angle/key/policy parity, not C# tests or timing.
At sequence512, a traversal of the whole current corpus in that independent guard simulation performs5100 legacy vs2690
optimized key constructions. This is an operation count on that traversal, NOT application wall-clock reduction.
Both policy simulation and dataset measurements are reproducible; exact .NET Unicode casing/serialization bytes are tested
only by the defined C# regression, not by Python's JSON serialization or casing.
