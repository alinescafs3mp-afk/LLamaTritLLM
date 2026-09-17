# Audit22 experimental method and runtime costs

The purpose is conversational transfer, NOT a throughput claim. Explicit category quotas bypass the old length-bucket
resampling for this opt-in path only. Padding and per-step work can INCREASE. Legacy plans use the unchanged old route.

The new planner owns integer indices/target counts and references existing encoded arrays. It does not clone token buffers.
The unused legacy phase-reference arrays are not constructed for a v22run. Every actual train step is still one forward,
loss and backward/update. RNG state remains session-owned and restored with the optimizer. A new requested course restarts
phase positions rather than silently resuming an interrupted job. Cancelled selection can consume RNG after its first draw;
worker recovery restores the last checkpoint. Pre-cancelled selection must consume nothing.

New read-only development generation adds up to8cases at existing bounded automatic checkpoints and62on explicit request.
It is CPU inference from committed packed weights, even when the training device isCUDA. This adds latency between intervals,
not a second training pass per step. Per-case feedback exists. Do not reinterpret sample counts as a full evaluation.

Independent Python A/B (same v22expanded data,initial seed42,821120parameter model,B8,6000steps):

```bash
python tools/transfer_learning_experiment.py --arm legacy --steps 6000 --output /tmp/v22-legacy.json
python tools/transfer_learning_experiment.py --arm balanced --steps 6000 --output /tmp/v22-balanced.json
```

Legacy arm uses v21reference-pool+length-bucket selection with NEW data in the general pool. Balanced arm uses explicit
per-batch quotas, original final facts grouped by final question, and starter+new language priority. This A/B changes
sampling strategy, not data or initial seed. It is NOT a pure v21package versusv22package comparison. Independent Python
RNG/initialization differ from C#. Guard evaluated every500steps; stop preserves last accepted weights, final output reports
actual stop reason. Heldout generation evaluated only after the selected run, not used for gradient or early stopping.

The helper limits generations to112bytes (128for open prompts); its decoder may replacement-decode an incomplete final
scalar. That is a disclosed limitation of the independent helper, NOT evidence of a production UTF8decoder failure.
Results retain every answer and per-family counters, not just selected successful examples. Native tests remain separate.

After real C# compilation, run in a new isolated parent directory:

```text
TritStudio.Trainer --learning-lab --course --context-practice --transfer-practice --enforce-guard --steps 6000 --batch 16 --lr 0.001 --output-dir <new-experiment-parent>
```

--cuda requires realGPU and must reportCUDA. Exit3 is a rejected candidate, NOT a completed learning course. Source authors
cannot claim native speed, memory or learning success from Python. A shorter disclosed pilot is acceptable for host debugging;
a full target run is still needed. Keep all previous benchmarks, trained export/learning smoke and runtime failure gates.
