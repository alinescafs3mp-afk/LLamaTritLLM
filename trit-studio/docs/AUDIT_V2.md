# Global source audit 2 · 2026-09-16

## Scope and evidence boundary

Audited the provided TritStudio_v1 source archive, then implemented this cumulative revision.
Runtime authoring environment has Python/PyTorch CPU but no .NET SDK/compiler, GUI execution or CUDA.
**C# build, C# tests, desktop rendering and GPU tests are NOT RUN here.**
All fixes below mean implemented source changes, not claims of executed C# behavior.
Grok's delivery gates and the target laptop checks remain mandatory.

## Findings and implemented repairs

| ID | Finding | Change / acceptance evidence to run |
|---|---|---|
| A2-01 | No single unambiguous Windows deliverable from a Linux build host | C# `tools/Delivery`, `scripts/deliver.*`, self-contained win-x64, CPU+CUDA, absolute receipt and SHA256. Host tests precede publication. |
| A2-02 | Buttons could leave the user uncertain whether a command was accepted or finished | Central `RunButton`, progress/stage card, elapsed timers, disabled duplicates, persistent error block, status/log receipts. Headless interaction tests added. |
| A2-03 | Command submission was not the same as durable queue acceptance / completion | ID-correlated accepted/completed protocol; `Request` waits for completion, queued/duplicate counts are visible; overflow returns explicit failed completion. |
| A2-04 | Stop arriving before task initialization could allow old work to start | Cancel-generation fence captured at enqueue and checked before/inside operation startup. Black-box immediate stop test. |
| A2-05 | Weights/checkpoint could disagree with separately stored data/settings | Checkpoint v2 stages and hashes settings, training and validation alongside model, optimizer, RNG, counters and online commit ledger. Rollback reloads the same snapshot's inputs. |
| A2-06 | Restart/rollback could replay work or lose pause state | Durable RuntimeFlags, authoritative mode-state events, commit/replay reconciliation, no automatic retry of rejected/rolled-back IDs. |
| A2-07 | Dataset support silently lost conversation context and could clip targets | Full role-validated multi-turn parsing, history-aware identity, prompt/history masking, complete target preservation, UTF-8-safe text splitting. |
| A2-08 | Training corpus was only a demonstration | 298 training, 40 control, 40 held-out examples; reviewed original content, schema/hash/disjointness checks, frozen-test policy. |
| A2-09 | Validation/test data could be selected as a training file | `LoadTraining` rejects records marked split=validation/test. Worker excludes the active validation identities too. |
| A2-10 | Padding and transient device tensors wasted memory/compute | Variable-sized cached examples, dynamic batch length, batched bucketed validation, RoPE/mask caches, deterministic parallel initialization/packing, explicit dispose scopes. |
| A2-11 | Reloading/resizing could overlap old/new device allocations | Dispose prior session before replacement; carry optimizer and sampler state, move optimizer state to selected device; partial-construction cleanup. |
| A2-12 | Source needed compile hygiene | Missing ModelFiles namespaces for SHA256/MemoryMarshal restored; all projects included in solution; executable scripts and pinned packages retained. Actual compiler gate still required. |
| A2-13 | UI could confuse failed nonmutating requests with stopped trainer | Durable mode-state events; failed input no longer unconditionally flips online toggle; create preserves the selected online preference after success. |
| A2-14 | Disk history could grow forever | Free-space preflight, 128-revision publication guard and explicit locked offline pruning retaining active and immediate parent. No live deletion. |
| A2-15 | Closing/reopening and CUDA startup could obscure progress or hang | Visible closing/opening/device states, one close sequence, bounded child shutdown/kill, actual CUDA tensor probe and logged CPU fallback. |

| A2-16 | Async UI test could bind to a non-awaiting headless overload | Use explicit `Dispatch<int>(Func<Task<int>>)` and a bounded test deadline after inspecting pinned Avalonia 11.3.22 API. |

| A2-17 | Long feedback could consume the whole window; approved-button bookkeeping retained removed chat controls | Bound feedback areas with full details in tooltip/log; keep one-shot state on the button itself, no permanent control set. |
| A2-18 | Intentional stop could be presented as a failed command | Explicit cancelled completion flag mapped to task cancellation and the UI cancellation state. |

## Executed checks in the authoring environment

`tools/validate_dataset.py`: hashes, schema, role order, UTF-8, exact and normalized duplicates, cross-split prompt overlap,
sequence budgets. Result in `reports/dataset-audit-v2.json`.
`tools/numerical_reference.py`: independent PyTorch vs NumPy math; real autograd toy fitting; fixture generation.
It does not import or execute this application's C# code. Result in `reports/numerical-reference.json`.
`tools/static_audit.py`: XML/JSON references, script syntax, source structure and safety invariants. This is not a compiler.
Package verification and patch reproduction are recorded in the enclosing cumulative package evidence.

## Residual limits / do not disguise them

The 378-record corpus is a starter, not proof of general fluency. Quality needs evaluation on unseen prompts after actual training.
Full FP32 training uses dense attention, not FlashAttention or mixed precision. CPU/GPU speedup was not measured here.
The fixed byte tokenizer spends context faster than a trained subword tokenizer. Default 512 training /1024 chat was chosen so bundled conversations fit fully.
The 128-snapshot guard requires explicit offline maintenance; live automatic retention is intentionally not implemented.
Raw chat and replay journals are local plaintext; excluding a message does not erase history or unlearn previous weights.
Binary model integrity is checked but files are not cryptographically signed. Workspace optimizer snapshots are for trusted local workspaces.
Atomic files and staged snapshots improve consistency, but power-loss durability on every filesystem is not certified.
No pretrained weights, original `.bin` importer, distributed training, remote API server, package signing or GPU VRAM telemetry is claimed.
