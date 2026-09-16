# Audit13 performance verification

## Execution boundary

Authoring evidence is Python specification/structure only. No C# compiler, native run, actual GUI or CUDA was available.
Run the existing delivery scripts and Check-laptop.cmd; they include the new attentionAndJournal benchmark section without
removing any previous section. The benchmark only creates synthetic temporary files/models, never trains user workspaces.

## CPU attention

Compare actual v12 time-major scalar value accumulation against CpuAttention.Vector<float> channels plus scalar tail.
Heads4/KV2, positions32/256/1024, head widths16/33/64. The odd width is a kernel edge test, not a valid RoPE model preset.
Identical preallocated buffers and values, same score Dot and softmax. Verify numeric outputs outside timing, warmup4 calls,
24 timed calls per shape/mode, both trial orders. Record target SIMD enablement/float lanes,elapsed,current-thread allocations.
No extra threads in either value kernel. Full-model native/managed incremental parity remains a separate mandatory gate.

No minimum speedup threshold. Vector gains depend on hardware/JIT/shape and extra checks; the optimized route may be slower
for some small shapes. Keep evidence, do not doctor the legacy baseline. Synthetic kernel speed is not whole-chat throughput.
Cancellation, invalid input and nonfinite tests are correctness work, not timed kernels. No zero-allocation process claim.

## Chat tail

Generate10000 valid static JSONL exchanges in an isolated temporary directory. Compare old bounded-start/forward-deserialize
reader with the fixed-end/newest-first reader, requesting250. Verify identical chronological turns BEFORE timing.
Warmup2,8 passes, both orders. Report ms,current-thread allocations,payload bytes and records inspected. The old implementation
really does deserialize all valid rows in this window. The new one still reads its byte window; it skips JSON object creation for
unneeded older rows.10000 versus250 parsed records is NOT a guaranteed40x speedup. Shared file cache and JIT affect timings.

Growth on first read, truncated/oversized/invalid UTF8 lines, exact boundaries and cancellation are managed tests, not this static
file benchmark. The cap bounds read payload, not pool retention or total deserialized-object memory. Captured length is not
an atomic filesystem snapshot. No persistent logs are truncated or compacted by this optimization.

## Actual reporting

Record runtime/OS/CPU, SIMD width and configuration alongside outputs. CUDA synchronization remains in the existing native
training timing; this added section is CPU-only even when invoked by a CUDA trainer. Do not relabel it a GPU improvement.
Retain all earlier textPreparation,cpuPreparation,onlineReplayPreparation,checkpointCorpus,packing,input-cache and neural sections.
Store the real benchmark JSON/logs in the normal delivery/diagnostics locations. Run all tests before reporting a successful ZIP.
