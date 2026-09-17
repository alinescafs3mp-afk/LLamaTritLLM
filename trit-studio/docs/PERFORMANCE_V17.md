# Audit17 measurement and learning acceptance

This revision prioritizes correct learning evidence and phase-correct memory planning over more speculative speed optimizations.
No measured C# speedup, GPU utilization claim, VRAM reduction or native leak-free result is supplied.

1. Retain every existing synchronized native benchmark. Run the6-reply native learning gate independently, not as a speed ratio.
2. Record initialization time/RSS/actual GPU allocation separately from training with B8/T512 and B32/T1024 ceilings. The dynamic
   corpus can use much shorter sequences. Log actual batch shape; do not label the ceiling as allocated length.
3. Reproduce the old conservative28342MiB formula using the experimental config. Zero-step initialization must not fail that
   training-activation quota. CPU actual forward/backward must STILL enforce configured host budget. CUDA is not RAM.
4. Run quality on an existing committed checkpoint, compare active pointer, hashes, optimizer counters, runtime flags and replay
   before/after. Only diagnostics output and nonpersistent performance caches may change on success.
5. The reference fixture has six learned responses and a recorded exact SHA256. Measure actual C# loading/generation only after
   verifying byte digest, context/reset, greedy parameters and expected replies. Unseen fluency is a separate experiment.

Python trial timing is incidental instrumentation on a different implementation; it is NOT native performance evidence.
