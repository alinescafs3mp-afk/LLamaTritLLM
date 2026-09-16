# Audit9 performance measurement

## Changes to isolate

1. **Snapshot corpus serialization**: cache exact immutable training/validation JSON, retain <=4MiB per slot;
   oversized entries stream once without retention. No second read for these corpus digests. Complete separate
   files and fsync remain. New output field `checkpointCorpus` in `--benchmark` records8 timed writes after warmup,
   cache hits/serialization count, bytes retained and current-thread managed allocation traffic. It uses a
   private temporary directory and512 synthetic records. All hashes are checked outside the timer.
2. **Batch statistics**: lengths and target counts prepared once per corpus. Preserved RNG/selection must pass
   before interpreting timing. Native end-to-end TrainStep benchmark retains its three existing variants.
3. **Changed weights**: exact bounded-parallel comparison for larger models, with cooperative cancellation.
   This is checkpoint telemetry, not an additional training update. No fixed speedup threshold is appropriate.
4. **Input failures**: UTF8/line bounds and cross-file state checks prevent continuing expensive work with
   malformed inputs. They are correctness/preflight improvements, not reductions in neural arithmetic.

## Running

Use `scripts/deliver.sh` (or deliver.ps1) on Grok's build host. It already runs CPU acceptance and the benchmark.
On the Windows laptop, use Check-laptop.cmd, which executes actual CPU/CUDA tests and reports. To repeat:

```powershell
.\trainer\TritStudio.Trainer.exe --benchmark --output cpu-v9.json
.\trainer\TritStudio.Trainer.exe --benchmark --reverse --output cpu-v9-reverse.json
.\trainer-cuda\TritStudio.Trainer.exe --benchmark --cuda --output cuda-v9.json
.\trainer-cuda\TritStudio.Trainer.exe --benchmark --cuda --reverse --output cuda-v9-reverse.json
```

The checkpointCorpus section runs managed CPU/disk work even in a CUDA benchmark. Do not label that section a
GPU kernel speedup. Cache mode still writes every byte and flushes; on a slow drive, disk latency can dominate.
Allocation traffic is not peak RSS.8MiB total retained corpus bytes excludes source arrays and temporary serializer buffers.
No published C# wall-clock result was measured in the source-author container. Independent Python reports are
math/algorithm checks and operation counts only.

## Required correctness invariants

Warm/cold cache emits byte-identical files and SHA256. Different corpus object invalidates, cancelled/error writes
never certify success, existing paths survive refusal, zero budget works, oversize uses one pass. Restore verifies
all required files. Bucket/uniform and forced-correction selections preserve all choices and sampler state.
Counter mismatch must be rejected before replacing the native model; active pointer remains old on failed rollback.
