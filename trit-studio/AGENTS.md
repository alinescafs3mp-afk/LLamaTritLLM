# Trit Studio implementation constraints · audit14

Keep C#, Avalonia two-tab UI, TorchSharp CPU/optional CUDA training in a separate process, and portable CPU-only inference.
The default Windows portable delivery includes both trainers. The Linux build host need not have a GPU.
Do not replace real model forward/gradient math with string lookup or canned replies. Online learning really changes master weights;
inference uses immutable published snapshots. No training on unapproved model answers. No claimed parameter-count growth during SGD.

Every user action must immediately acknowledge, show progress/stage and end in visible success/error/cancellation. Never silently
truncate targets, lose a new draft, report queued data as already learned, or let an uncommunicative trainer run invisibly.
Preserve workspace leases, rollback ancestry, high-water revision IDs, coherent checkpoints and explicit offline retention.
Raw chat/replay are local plaintext; do not commit them, credentials, checkpoints or compiled binaries. Exclusion is not erasure/unlearning.

Use the pinned dependencies unless a verified compatibility issue requires a documented change. Model binary format remains V1.
New optimization must keep a reference/parity test; native allocation lifetimes require train/eval/copy/reopen soak.
No GPU, UI, performance or compilation claim without its executed log. Hash checks are not signatures or durability certification.

Corpus version is conversation-ru-v14. Every future audit adds reviewed original coverage and retains all prior blind files.
Never train on control/test/challenges to increase scores. Keep an existing workspace validation set unchanged across upgrades.
All normal defaults and artifact receipts must remain simple enough that the owner hands Grok one folder and receives one usable ZIP.

Read docs/AUDIT_V14.md and docs/ACCEPTANCE.md. Authoring C#/native/UI/CUDA tests are NOT_RUN; real execution is the release gate.

Freeze pretrain.jsonl byte-for-byte across routine dataset audits. A deliberately changed baseline requires a separately named/versioned experiment, never silent replacement. Compare historical validation scores only for the same effective sequence budget. Keep completion payload validation before request removal and prospective restore before active pointer publication.

Keep corpus arrays immutable after publication; a replacement array invalidates serialization/evaluation caches. Never mutate nested histories in place. Strict imported text is UTF8, optionally BOM-prefixed. A generated answer and its durable journal write are different outcomes; report each accurately. Check counter/ledger semantic agreement before activating a restored snapshot.

Outgoing pipe deadlines bound transport only, not optimization time. Ambiguous transmission is terminal: fail pending receipts, stop child, no auto-retry.
A nonmutating preflight refusal must not reload/upload/re-publish the unchanged model. Any later RNG/optimizer/weight/settings mutation retains full recovery.
On runtime-mode persistence failure disable in-memory online work; never claim a disk update succeeded or that reopening cannot restore an old flag.

Workspace switching must verify new bounded conversation state and packed model/history before old-client teardown. Keep old+new
CPU model overlap explicit; do not claim a preflight-memory saving. Reuse verified preview objects rather than hash/unpack again.
The initial-publication master shortcut owns an immutable constructor WeightSet; invalidate BEFORE every native update/restore,
and respect Model.WeightVersion. Never publish old random weights after training. No direct in-place mutation of owned arrays.
