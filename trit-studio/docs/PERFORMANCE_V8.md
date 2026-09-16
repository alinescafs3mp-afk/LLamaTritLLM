# Audit8 performance measurement contract

There is no measured C# wall-clock speedup or native peak-memory result in the source-author environment.
This revision removes repeated CPU preparation of validation packets and a full replay scan from each manual training operation.
The bounded cache trades at most 16 MiB of retained INPUT/TARGET/POSITION array payload per training session for repeated preparation.
It does not store CUDA tensors and does not replace validation after a weight update. The older validation-result cache still skips an identical
model/corpus pair, while the new preparation cache helps when results MUST be recomputed. UI telemetry distinguishes these two counters.

## Run actual tests and measurements

`TritStudio.Trainer --benchmark --output report.json` runs isolated synthetic work and never opens owner model workspaces.
Use `--cuda` only on a working CUDA runtime; use `--reverse` for the alternate benchmark order supported by the executable's existing parser.
The standard delivery and Check-laptop wrappers already invoke the benchmark; `--reverse` selects the alternate trial order.

The new `validationPreparation` section measures zero-retention vs retained-cache preparation on the same 128 synthetic examples.
Warmup occurs first, 16 complete passes are measured, every target is counted exactly once per pass, elapsed time and current-thread managed
allocation traffic are recorded. Retained array bytes are reported separately. It is NOT neural-network compute or GPU timing.
The existing native whole-step/eval, portable inference allocation and legacy-vs-current pack benchmarks remain required.

Repeat on both CPU and RTX5080, both trial orders, same architecture/context/batch/data and warmup. Capture several runs with median/range.
Do not turn a relative timing ratio into a pass assertion; tiny models and host load can reverse the performance ranking.
Review actual training speed, responsive controls, RSS growth across repeated reopens, and optional external GPU monitoring.
RSS is not VRAM; a caching allocator retaining a stable pool is not evidence of a leak, and one short successful run is not proof of leak freedom.

## Correctness and capacity gates

Compare cached/uncached inputs, targets, flat positions and counts with 0, tiny and full budgets, short final batches and ignored trailing padding.
After a real optimizer step the evaluated loss MUST be recomputed; preparation counters may show reuse. On corpus/resource/session replacement,
no arrays or losses from the previous effective inputs can be reused. Invalid/cancelled preparation must not become successful cache entries.
Keep finite loss/gradient, SDPA/explicit, selected/full projection, optimizer resume and snapshot tests unchanged.
The independent Python `audit8_reference.py` reports data/algorithm equivalence only. It is not a substitute for the C# benchmark.
