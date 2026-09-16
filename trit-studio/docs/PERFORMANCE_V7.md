# Performance audit7: measurement contract

Cumulative baseline: all v6 pack/hash/batching/projection/SDPA optimizations remain.
New changes: native SiLU instead of sigmoid/multiply composition; bounded byte output storage instead of integer-list plus repeated
intermediate decode arrays; immutable validated sampler settings do not allocate per token; teaching context is captured once rather
than reconstructed at each delayed approval. Staged initialization does not encode the full conversation-training set in random/basic mode.

The common validation set is still prepared to keep comparison/acceptance stable. Zero-step creation still opens training machinery.
No measured C# startup/throughput/VRAM number is supplied. The Python reference tests values/gradients and output-storage invariants.

Run the existing trainer --benchmark on host CPU and target CPU/CUDA; preserve warmup, synchronized timing and JSON evidence.
Also record separately: create mode0 time/peak RSS; create mode1 with a fixed step count; pure basic->conversation transition; stage-copy
bytes/time; generation allocations for the same saved snapshot, prompt, context, thread count and greedy sampling settings.
Do not include console printing/UI rendering in a kernel timing and call it a model throughput comparison.

For any claimed improvement compare the same weights and inputs against exact v6 in separate processes. More trained weights or a changed
corpus is not a fair performance baseline. Allocator cache retention is not automatically a leak; require repeated bounded-workload soaks.
No mandatory speed ratio is used to excuse incorrect outputs or skip lifecycle checks.
