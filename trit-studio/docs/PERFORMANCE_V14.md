# Audit14 performance verification

## Boundary and execution

Authoring evidence is independent Python math/policy and structure only. C#, native training, GUI and CUDA are NOT_RUN here.
Run normal scripts/deliver.* and Check-laptop.cmd. They invoke the extended benchmark automatically. No owner model or chat is
opened by the benchmark; it creates an isolated small random model. Retain ALL earlier benchmark sections and correctness gates.

## initializationAndElementwise: initial publication

Compare Model.CopyMaster (real FP32 device->host copy) with CapturePublicationMaster's verified eligible initial-owner reuse.
Warm up each path; time five calls in both orders. For CUDA synchronize before/after timing and refuse CPU fallback. Compare
all returned float values outside timing. Record elapsed and current-thread managed allocations; the modeledReturnedFloatBytes
field counts newly returned float payload, NOT measured peak RSS/VRAM or PCIe traffic. Native allocator pools can remain allocated.

This shortcut is valid only before weight update or state restore. Test identity initially, pre-cancellation, actual native-copy
parity, Model.MarkUpdated, state restoration, failed restore and fresh capture after a real gradient step. Do not speed up the
benchmark by using the shortcut after changing weights. No snapshot packing/disk-write/optimizer-save costs are included in this
microbenchmark; do not present it as total create-time or total train-time speedup. Creating random weights still initializes native
machinery. The old original WeightSet already exists; no new whole-model cache is created just for the shortcut.

## Channel-wise CPU operations

Compare scalar reference and portable Vector<float> RMSNorm scaling followed by residual addition for widths16,64,128,256,512.
Use identical inputs, prior Dot reduction and multiply order, preallocated outputs and no extra worker threads. Each iteration
normalizes into the output anew; do not repeatedly add into a growing accumulator. Warmup20,1000 timed iterations, both orders.
Verify RMSNorm and residual outputs before timing, then record hardware vector flag/float lane count,elapsed and allocations.
Use tolerances, not mandatory bit equality across JITs. Tiny shapes can be slower. No fixed speedup pass threshold.

Null/bad shape, exact-array aliases, empty residual, cancelled writes and nonfinite reduction are separate managed contracts.
Retain full managed incremental/native Torch parity and scalar attention regression. A Python vector chunk matches a mathematical
specification but is not evidence of .NET intrinsics, JIT scheduling, CPU instructions, cancellation races or native lifetimes.

## Workspace preview

Read/verify/unpack the destination once while keeping the old CPU model live. Reuse those exact objects for activation instead
of reloading them. Verify failure preserves the old session before measuring IO. This is primarily a correctness improvement;
old+new model arrays temporarily coexist. Do not claim peak-memory savings. A newly started trainer still fully verifies its
own recovery state; that is not redundant IO to remove. Native restore failures remain visible with CPU-only fallback where valid.

Record build/runtime/CPU/GPU/OS, test logs and both-order benchmark JSON in normal delivery/diagnostics locations. CPU fallback
is not a CUDA pass. Actual host tests must pass before returning the fresh Windows portable ZIP to the owner.
