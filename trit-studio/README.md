# Trit Studio · audit16

C# / Avalonia desktop model laboratory, TorchSharp trainer, independent managed CPU inference. Training from random weights;
not a bundled pretrained assistant. UI in Russian. See README_RU.md and docs/AUDIT_V16.md.

Current delivery: app-only update by default. `bash scripts/deliver.sh` (or scripts/deliver.ps1) runs all real host acceptance,
builds both trainer entrypoints and writes artifacts/TritStudio-win-x64-update.zip. Existing vendor runtimes must match the
REQUIRED_RUNTIME_FILES manifest. Full installation only via explicit --full. Source handoff is not a compiled package.

Selected model in header controls chat AND training. Architecture creation is separate. Optional staged evolution, explicit
online learning/corrections, persistent snapshots and clear-chat boundary are retained. Model deletion moves local folders to
app-local trash. Clearing context does not delete history, discard training queue or unlearn weights.

Authoring checks here do not run C#, Avalonia or CUDA. Grok must compile/test and target must run Windows/RTX5080 acceptance.
All upstream MIT attribution preserved. Binary format is Trit Studio V1; byte vocabulary262. Corpus is synthetic starter material.
