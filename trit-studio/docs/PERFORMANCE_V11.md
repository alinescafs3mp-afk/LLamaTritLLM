# Performance acceptance: audit11

No C# runtime benchmark was executed by the source author. Report executed evidence separately from inferred work reduction.
Use the existing build/delivery and Check-laptop.cmd paths. Retain all earlier model/math/serialization/textPreparation benchmarks.

## onlineReplayPreparation

The C# benchmark constructs256 base examples,64 retained examples with history,4 pending examples and a separate control.
Legacy v10 validates and encodes all64 retained rows; optimized path uses the same logical combined array indices and prepares
only selected retained entries. Test steps0,254,256,319: all-base, crossing to learned, learned-only and wrap to base.
Verify every input and target array outside timing; preserve pending order and selected duplicate references.
Warmup3 per variant,32 measured repetitions, both orders through the existing --reverse harness. Record elapsed time,
GC.GetAllocatedBytesForCurrentThread and pool positions. This isolated managed test excludes neural training, disk and UI.
No pass/fail timing ratio. Baseline encoding runs one thread, matching v10's small-array encoding policy.

## Unchanged manual corpus preparation

Reuse the training owner only for exactly equal ordered immutable records and unchanged sequence budget. Compare cold and warm
manual operations with the same material. Changing content/history/source/order must invalidate. Changing sequence budget must
re-encode training AND control. Reusing prepared control arrays must NOT reuse a loss result across weight updates.
No claim that first-time imports or new corpus versions avoid encoding. Record comparison still costs O(number of records).

## Work and failure boundaries

Selected learned encoding count is0..4 instead of up to64 per micro-update. Do not translate that into16x total training speed:
four gradient steps, validation, master transfer and checkpoint IO still exist. Validation/candidate safety checks are preserved.
Snapshot capacity preflight can save an otherwise doomed long run; it is not faster neural math. Verify exact boundary fit and
refusal before weights/publications. Do not bypass128 limit, silently increase publish interval or auto-delete snapshots.

## Stability and UI

Repeat online updates, mode/LR edits, model changes, stop and reconnect while measuring actual RAM/VRAM on target.
No late edit from another workspace may be sent. Pending acknowledgements visibly disable incompatible actions.
Protocol frame tests must show process exit before parent disposal, not merely a UI timeout or a fake task completion.
Malformed input may cancel unpublished work; document which verified revision remains. Keep complete failure logs.
CPU allocator pools and CUDA caching are not themselves proof of leaks; short runs do not prove absence of leaks.
