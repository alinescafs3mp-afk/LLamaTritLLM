# Trit Studio audit14

C# desktop laboratory: Avalonia UI, TorchSharp CPU/optional CUDA training, independent managed CPU inference.
This is a cumulative SOURCE handoff, not a compiled application or pretrained weights. The author has NOT executed
C# build/tests, TorchSharp, Avalonia or CUDA. Grok must run all real build/runtime gates before delivering the Windows ZIP.

## Staged experiment

1. Create without training (UI default): random weights, zero optimizer steps, online learning OFF.
2. Basic pretraining: 48 separate, frozen short texts; no conversation/custom/replay targets.
3. Explicit conversation training: 2474 seed examples plus previously attached basics and optional selected custom files.

The explicit one-click create+conversation alternative remains. Normal optimization changes values, not parameter count.
Manual stages leave online learning paused for comparison. Enable it explicitly when ready. First successful stage references
under workspace/evolution are inference-only and survive checkpoint pruning. Resume training from the full workspace.
See docs/EVOLUTION_RU.md. All old stage references and existing workspace validation remain unchanged on software upgrade.

## What audit14 changes

Read-only workspace preflight preserves the live model/client/draft on malformed destination data, then reuses the verified model
and history for activation. First zero-step publication reuses the original immutable master instead of copying it back from the
native device; every update/restore disables that shortcut. Portable vector channel operations cover RMSNorm scaling and residual
addition, keeping the same reduction, scalar tails and CPU independence. Model read pre-cancellation precedes file access.
The new initializationAndElementwise native benchmark compares real paths. No C# speed/memory gains are measured here.

## Corpus and privacy

2474 training + 328 validation + 240 held-out + 48 separate baseline = 3090 records. Largest full sequence:453 byte tokens.
The 48 basics and all twelve previous held-out files are byte-frozen. Synthetic starter data is not a fluency guarantee.
User messages/replay are local plaintext. Exclusion prevents the supported teaching route, not disk storage or unlearning.
Unapproved model replies are not automatically correct training answers. All prior dataset isolation guards remain.

## Delivery

From this directory run `bash scripts/deliver.sh` (Windows: `./scripts/deliver.ps1`). Default target is win-x64 CPU+CUDA,
even on a CPU-only Linux build host. The owner has installed build prerequisites; do not reinstall or alter drivers gratuitously.
Fresh result: artifacts/TritStudio-win-x64-portable.zip, absolute path in WHERE_TO_PICK_UP.txt and DELIVERY_RESULT.json.
Copy the WHOLE ZIP to the laptop; Check-laptop.cmd then Start-TritStudio.cmd. No .NET SDK is needed on the target laptop.
Cross-publication does not execute Windows/CUDA. CPU fallback is never a CUDA pass. No stale ZIP may be reported as fresh.

See GROK_START_HERE.md, docs/AUDIT_V14.md, docs/PERFORMANCE_V14.md and docs/ACCEPTANCE.md.
The owner fork retains original console sources/notices. Trit Studio V1 model files are not the upstream .bin format.
