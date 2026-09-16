# Audit15 completion plan

Read GROK_START_HERE.md and the enclosing source preflight. Source handoff complete; C#/UI/native/CUDA acceptance must run
on Grok host and laptop. Preserve deployed audit14 integration fixes. Default target win-x64 app-only UPDATE, no vendor runtime
payload. Compile both trainer entrypoints and update every adjacent corpus; keep pins unchanged. Never silently choose --full.

Run scripts/deliver.*. All managed/UI/worker/native tests and existing benchmarks precede output. Add native trained-export parity,
real modal clear/cancel/failure and Windows model-switch/trash/DPI acceptance. Generated update must be applied to a COPY of the
old compatible installed folder and pass Check-Update. Missing runtime is a compatibility problem, not license to download or
bundle gigabytes. Do not overwrite user models. Final report: exact path, checksum, commit, passes and remaining target checks.
