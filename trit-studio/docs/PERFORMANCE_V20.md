# Audit20 performance semantics

The priority is useful conversation learning, not maximized tokens/s. This iteration intentionally changes
an opt-in training distribution and loss. Do not advertise a throughput gain from it without measurements.

Assistant-prefix supervision adds1320views on the current3247source course rows. Shared replay pools add
references only; they do not duplicate encoded token arrays. The full unique union is checkpointed once.
Short-first phases reduce early sequence lengths but later phases use all prepared data. Those stage times
must be measured separately and not compared as identical workloads.

Equal-example CE allocates one float vector per supervised batch and uses unreduced CE plus weighting.
It reuses existing logits, retains all context attention and does not add a second forward. Control loss/
accuracy remain full original token-mean evaluation; memory/failure/cache/version invariants remain.

Conversation checks intentionally perform CPU generation on saved weights. At normal course phase/last
boundaries there are at most4attempts; failures are warned, not retried on every subsequent checkpoint.
They can add noticeable time and temporary CPU model memory. UI reports the phase so this is not a hidden
training stall. No per-step report writes. Explicit manual probe can be repeated separately.

Retain ALL previous native performance sections and measurement methodology. Re-run train/evaluate/export/
reopen soak after the objective changes; managed metrics are not peak VRAM. No new dependency/library version
or GPU-only inference path was added. Author native runtime/GUI/GPU execution is NOT_RUN.
