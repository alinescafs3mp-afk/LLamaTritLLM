# Audit18 performance verification

Source author cannot run C# or CUDA here. Report only actual target measurements, never convert Python math into a speed claim.

Accuracy reuses loss logits. Measure native train/evaluation latency and RSS/VRAM with the same model/data/settings before/after
this change; synchronize CUDA outside timed loops. Do not count UI display refresh as a training improvement. Argmax introduces
an extra scan over supervised logits but no extra forward or per-step disk write. Native benchmark's existing measured steps
include this cost. Add metrics fields to its output for inspection; no enforced speed ratio.

QualityProbe reuses accuracy from its first Evaluate(sample) instead of evaluating each sample individually a second time.
Retain greedy answer and native/CPU logit comparisons. Verify same supervised count and numerical accuracy, excluding warmup
and fsync/log output when isolating compute. Whole-control display is independent of the diagnostic sample.

Next-run editor autosave writes a small bounded JSON after450ms inactivity, not per keystroke or per training status. Verify
status/Ready traffic causes ZERO extra draft writes. Invalid typed text must not silently save old values. Frozen editor control
arrays avoid allocating new arrays on every status UpdateState. Test slow/unwritable storage and model switch/close boundaries.

Batch64 is an allowed resource ceiling, not an optimal default. Compare batch8/32/64 at fixed total target exposure AND separately
at fixed steps. The latter processes more examples; do not call it a controlled quality comparison. Retain phase-specific RAM
checks and actual CUDA allocation errors; no guessed RAM-to-VRAM conversion or automatic LR scaling.

Keep all existing benchmark sections and six-reply trained-export acceptance. CPU fallback is never a CUDA pass.
