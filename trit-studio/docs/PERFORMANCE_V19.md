# Audit19: measure learning, not just kernel throughput

This iteration is not a measured C# speed optimization. Independent Python CPU experiments are in
reports/learning-investigation-v19.json, including seed/control/baseline hashes, versions and raw replies.
The primary result is reproduced degeneration under constant.01, not fluent completion at.001.

## Mandatory real experiment for Grok, opt-in for users

Run all existing delivery tests first. With a published HOST trainer execute:

```bash
/path/to/host/TritStudio.Trainer --learning-lab --steps 300 --batch 16 --lr 0.001 --output-dir /path/to/reports/learning-lab
```

Optionally compare an identical fresh.01 arm, then an opt-in `--scheduled` arm. Record initialization,
corpus hashes, all free outputs and real exit codes. The lab uses experimental4.79M configuration with
context1024/training limit512, writes unique run directories, takes no workspace path, and never changes
user weights. Maximum5000steps per experiment. Every100steps it measures full control, exports a packed
file and runs3empty-context greedy queries through the production CPU generator. Fixed public queries are
not extra training data. Corpus contains only bundled training, never controls/test targets in gradients.

The laboratory deliberately OBSERVES trajectories without the production validation-rejection guard. It
is a scratch experiment, not an alternate trainer for user workspaces. CLI exit0 means completion of the
experiment, not that conversation quality passed. `--cuda` must fail instead of falling back. Ctrl+C
cancels between operations. Artifacts may use disk space; no existing run is overwritten/deleted.
It is not invoked silently when the user opens the desktop or runs a quick check. No library update needed.

## Native correctness gates

LearningPlanSelfTest verifies actual optimizer group LR per step, persisted last rate, and explicitly
new-run offset after restore. CheckInitialRouteParity compares direct conversation and zero-create/reopen/
conversation through REAL Worker processes: same corpus, control, scheduled updates and final CPU master.
Existing fixture recall/learning-smoke/trained-export parity remain mandatory. Keep all previous benchmarks.

## Cost/limitations

Optional schedule is constant-time scalar host work per update. It adds no model pass. GenerationHealth is
small bounded-output Unicode counting; it does not modify text. The byte-bigram allocation is262*262*8bytes
plus marginal counts and predictions; explicit QualityProbe streams encoded records into it, no full third
encoded-corpus cache. Existing diagnostic/eval allocations still exist. No per-step automatic language
model evaluation is introduced. The auto snapshot policy trades fewer writes for longer recovery gaps and
less frequent validation, disclosed in UI. No claimed wall-clock/GPU memory improvement.
