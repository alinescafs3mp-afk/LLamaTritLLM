# Audit21 costs and experiment reproduction

This release is NOT a measured runtime speedup. Mixed curricula keep full-distribution supervision present
at every phase. Pools reuse EncodedExample arrays, with4N/2.5N/1.75N reference slots; token arrays are not
replicated. This increases reference-storage and changes training sampling. Old backend/batching/gradient
and publication guard remain. No additional optimizer steps are performed by conversation checks.

The new paired check uses CPU inference even if training isCUDA. Up to4pairs at bounded automatic course
checks,16manual; each pair contains two fresh conversations and capped64byte answers. It adds elapsed time
at those checks, not a second forward pass for every gradient step. Existing10-case checks are retained.

Independent reproducible experiments, from trit-studio:

```bash
python tools/context_learning_experiment.py --steps 3000 --every 250 --output /tmp/context-ab.json
python tools/context_learning_experiment.py --steps 6000 --every 500 --only v21_context_mixed --output /tmp/context-full.json
python tools/context_recall_report.py --weights /tmp/context-full-v21_context_mixed.pt --output /tmp/context-known.json
```

Both A/B arms have identical NEW data and initialization; only the phase pools differ. This is not a pure
v20-package versusv21-package quality comparison. Real stop guard runs at each evaluation; rejected candidate
is restored to the previous accepted state and the run stops. Training uses medium821120parameters,B8,CPU.
Final outputs include every public paired variant and open prompts, including failures. No user models used.

An optional independent --initialization small-residual trial tested smaller weights; it did NOT solve
unseen context and is NOT adopted in the application. Its raw result is retained, not hidden.

Native counterpart after actual compilation (isolated experimental output only):

```text
TritStudio.Trainer --learning-lab --course --context-practice --enforce-guard --steps 6000 --batch 16 --lr 0.001 --output-dir <new-empty-experiment-parent>
```

Use --cuda only with realCUDA and record reported device. Exit3 means the guard rejected a candidate;
earlier exported experiment files remain, rejected candidate is not exported. The lab is not a resumable
optimizer workspace. Exit0 means the requested experiment completed, NOT conversational acceptance.
A pilot can be shorter if declared; do not advertise it as the full course. No fixed throughput threshold.

Author has no .NETSDK/native runtime. All existing real benchmarks and correctness gates still need Grok/
laptop execution. Do not infer target speed or VRAM from independent Python elapsed time.
