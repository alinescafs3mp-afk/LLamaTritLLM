# Trit Studio audit15: compact UI, model selection, app-only updates and clear chat

## Evidence boundary

Completed source handoff of the previously unfinished audit15. Integration base is the owner's actual audit14 commit
`fa9c9261fdcf3f463855ab5fb2687e71ba4654d9` on `alinescafs3mp-afk/LLamaTritLLM`, branch `master`.
Its `trit-studio` tree was reproduced locally as `11d2b7fc6eb4c272c4ca0a443a12cbfa6e6ff7bf`.
Preserved Grok's real host repairs: TorchSharp mean(long[]), malformed-frame IOException, Linux EOF fixture duplicate handles.
No owner checkout or remote was modified. Previous original-base cumulative remains a valid source comparison, not the current deployed baseline.

Author ran data/structural/independent numerical and package reproduction checks. **NOT_RUN here: dotnet build, C# tests,
Avalonia render/headless events, native TorchSharp, Windows, CUDA and actual update-on-laptop.** The authoring container has
no .NET SDK and official SDK endpoint DNS fails. Do not convert source-test counts, Python passes or screenshot impressions
into compiled/runtime passes. Grok must run retained gates and repair genuine incompatibilities without softening assertions.

## Source changes and required regressions

| ID | Finding / owner requirement | Implemented source and acceptance |
|---|---|---|
| A15-01 | Full CPU/CUDA portable delivery repeated gigabytes of installed native libraries | App-only update is DEFAULT. Include our assemblies/apphosts/resolution metadata for App/Core/Runner/Tests and BOTH trainers; no vendor libraries. Explicit `--full` is exceptional, not a fallback. Exact retained-runtime manifest plus updated-payload digest. 64 MiB uncompressed update guard. |
| A15-02 | Each trainer and tests load their adjacent data, not just the exported root dataset | Include current data in trainer/data, trainer-cuda/data, checks/data and reference.json. Do not mistake old adjacent corpus copies for vendor dependencies; mixed-version bundles must fail. |
| A15-03 | Small update could falsely work with an unrelated CUDA Toolkit/runtime installation | Hash and size all vendor files from the same publish; Check-Update checks existing files without downloading/deleting. Pinned package versions are unchanged. Newer .NET patch dependencies may require reproducing the v14 runtime publish, NOT silently shipping a giant ZIP. |
| A15-04 | Tall always-visible status/sidebar/teaching controls overwhelmed chat | Compact Fluent density, smaller padding, collapsible default-hidden sidebar, compact persistent outcome/errors, progress only during work, lazy teaching controls. Retain draft/context/selection and visible accepted/in-progress/terminal states. |
| A15-05 | DPI and long settings columns had poor fit | Persisted 50/65/80/100/125% layout scale, default80% with compact controls. Responsive header and single-column training layout at constrained width. Real laptop DPI/popups/tab/scroll/keyboard checks remain mandatory. |
| A15-06 | No clear single model target for both tabs | Header model catalog + switching using verified existing open paths. Selector displays ACTUAL active model while candidate prepares. Read-only active architecture; separate new-model editor. Clear model-specific selected custom files after a successful switch, not on refusal. Inference files cannot train. |
| A15-07 | Model removal must not touch upstream or arbitrary external folders | Typed-name confirmation, direct managed-child validation, no root/path links, exclusive UI+trainer leases, same-volume directory move to app-local trash with restore receipt. Inactive removal preserves active model; external paths are forgotten, not deleted. No automatic permanent erase. |
| A15-08 | Old giant cards gave weak evidence of training state | Active step/revision/loss/config summary; explicit publication interval; redacted diagnostic report captures architecture, published/live counters, saved/next options and sampling, not chat contents. Model display name remains included. |
| A15-09 | Owner requested visible clear-chat action | Confirmed `Очистить чат` beside composer. Persist fresh conversation ID BEFORE clearing UI. Clear current visible history/context only; preserve draft+privacy, worker, sampling, online settings, queue, weights and journal. Failed state write preserves current conversation. Reload filters to new conversation so cleared rows do not return. |
| A15-10 | Existing portable numerical smoke compared initial tensors, not all trained disk stages | Extend real trainer self-test AFTER actual gradient steps: device logits vs trained packed disk/CPU incremental, identical in-memory-vs-disk quantized arrays, deterministic response parity across roundtrip. Not proof of fluent speech. |
| A15-11 | Dataset needed easier language as well as long constraint examples | v15 adds160 training,24 control,16 challenge, including short natural responses, grammar agreement and connected texts. All13 old blind files and48 basics byte-frozen. The previously shared v15 seed file is unchanged. |

## Safety and semantics

An update is NOT a standalone installation. Merge its CONTENTS into the existing compatible v14/v15 folder while the app and
trainers are closed. Never replace/delete the whole `trainer-cuda` directory, use mirroring deletion, or infer LibTorch from
nvidia-smi success. App `.deps.json` and `.runtimeconfig.json` are owned update payload and must match updated assemblies.
Dependency hashes are integrity checks, not signatures. Check-Update can read large installed DLLs; no repeated library transfer.
Full self-contained publish still occurs in the build staging directory to determine the exact file set. This can consume
build-machine bandwidth/disk/cache; the DELIVERY ZIP does not contain those vendors.

Model trash preserves payload and does not free occupied storage. It is local app trash, not the OS Recycle Bin. External paths
are unlisted only. All directory/file operations assume trusted local storage, not an attacker concurrently replacing every path.
Failed removal of an active entry attempts reopen but may require user reconnect; never report that failed operation succeeded.

Clear chat starts a new conversation boundary, not deletion/unlearning. An already queued training example may still execute when
online learning is enabled. Excluding a draft is preserved. Raw logs/replay/trash remain plaintext. The old log is retained for
manual inspection, but there is no new full-history browser. Inference-only clear is memory-only; it changes no model file.

Changing architecture creates a DIFFERENT model. Ordinary training changes existing values, not the parameter count. Raw and
48-text basic stages remain distinct; direct creation with conversation training is a supported route, not misuse. The owner
reported incoherent text from a 4.79M model, but no workspace weights/logs were supplied: no specific cause, step count, CUDA
pass or training fix is asserted from that screenshot. r12 is a revision, not the optimizer-step count. Use actual diagnostics,
training/control losses, deterministic known/new prompts and the trained-export check before blaming settings or data alone.

## Real acceptance (defined, not executed by author)

Run all managed cases (count in current static report), all retained child-protocol/worker/native suites and new Audit15UiChecks.
The latter exercises real confirmation buttons for clear/cancel, failed storage preservation, reload boundary, lazy teaching,
scale/sidebar feedback, failed/successful file switch, inactive managed removal and external forget. Native self-test must include
trained export parity on CPU and real CUDA. Core tests verify update allowlist, path/link/lease refusal and trash preservation.
Add actual Windows tests of rename with lease handles, dropdowns at50/80/100/125% and DPI100/150/200%, train/model switch gating,
startup in an existing workspace, and direct conversation creation. No screenshot/render pass is claimed without real rendering.

The package rehearsal separately proves source application/refusals, not these application semantics. C# source still needs
compilation and real native lifetime/leak checks. Host success precedes the fresh win-x64 UPDATE zip; laptop checks follow merge.
