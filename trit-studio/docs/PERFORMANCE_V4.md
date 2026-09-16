# Performance methodology: audit4

## Defaults and opt-outs

Advanced training options expose **SDPA** and **length bucketing**, enabled by default and serialized with workspace resources.
Uncheck SDPA to use the explicit causal reference path. Uncheck bucketing to restore uniform mixed-length batches.
Settings take effect at the next training operation boundary. Online updates deliberately keep correction/replay mixing.
No automatic dtype change or unchecked CUDA kernel replacement is introduced. Portable CPU inference stays independent of TorchSharp.

## Measure the whole job, not only a kernel

The normal UI displays target tokens/s, actual batch x effective sequence, padded fraction, validation time and result-cache hits.
A reused unchanged validation shows 0 ms and increments a cache-hit counter, not a fake new quality score.
Do not compare token rates without model shape, batch, sequence-length distribution, hardware, dtype and validation/publication cadence.
Byte-token counts are not words; particularly not Russian words.

`TritStudio.Trainer --benchmark [--cuda] [--reverse] [--output FILE]` creates temporary synthetic models in memory.
It does not open any owner's model, train on blind test files, or publish weights.
Three warm-up steps are excluded; sixteen measured steps per variant are timed with CUDA synchronization before/after the interval.
Variants use identical initial weights/data but different batching; their final losses are not an apples-to-apples convergence claim.
Reports include padded fraction, token throughput, full and cached validation duration and host RSS. RSS is not VRAM.
Changing the trial order and repeating on an otherwise idle laptop reduces warmup/governor/allocator bias. No speed ratio is hard-coded as a pass.

Host: `scripts/deliver.sh` includes an actual CPU benchmark report in the build-log directory.
Laptop: `Check-laptop.cmd` writes CPU/CUDA benchmark JSON next to its diagnostic logs. Nonzero test/benchmark exit fails that check.
Optional manual reversed CUDA trial after extraction:
```powershell
.\trainer-cuda\TritStudio.Trainer.exe --benchmark --cuda --reverse --output .\diagnostics\cuda-reverse.json
```
The source author did NOT run these C# commands. The independent `tools/performance_reference.py` checks math and shape accounting only.

## Required soak, not yet certified

On the real native build, run repeated train/evaluate/restore and open/close cycles. Track host RSS plus device memory with a GPU tool.
First establish allocator warmup baseline; retained allocator pools alone are not proof of leaked live tensors.
Test cancellation and restore failures. Every no-grad cache lease must end on success and exception; no cached tensor may cross a weight update.
Verify the fallback path and compare logits/gradients before enabling optimization on any unsupported platform.
Record C# build/API fixes and rerun all gates. Never replace real arithmetic with fixture/canned outputs.
