# Trit Studio audit16: second pass on v15, keyboard routing and safe recovery

## Evidence boundary

Complete source revision of the supplied v15 cumulative. Source-only handoff, not a compiled Windows build.
The owner fork was read again: alinescafs3mp-afk/LLamaTritLLM, master, fa9c9261fdcf3f463855ab5fb2687e71ba4654d9
(integrated audit14). No owner checkout or remote was changed. All Grok host fixes from that baseline are retained.
The wrapper accepts exact v15, exact deployed14, or the original additive base. Other owner changes require reviewed merge.

Executed here: actual data generation/hash checks, independent NumPy/PyTorch numerical specifications, source/script structure,
and package-byte reproduction/refusal in synthetic local Git repositories. **C# compilation and tests, Avalonia input/rendering,
TorchSharp native execution, PowerShell, Windows and CUDA are NOT_RUN.** dotnet is absent; official SDK access/download failed.
A Python comparison does not execute the changed C# implementation or certify its scheduling, memory lifetimes or Windows behavior.
Previous audit reports are historical. Binary format V1 and fixed 262-ID byte tokenizer are unchanged.

## Findings and implemented changes

| ID | Finding | Source repair and real regression |
|---|---|---|
| A16-01 | v15 STILL used the ordinary bubbling TextBox.KeyDown handler despite the owner's Enter report | Tunnel KeyDown intercepts plain Enter before multiline default handling; KeyUp/focus-loss latch prevents held-key repeat; modified Enter remains with the editor. Respect disabled Send and show why it was not sent. Actual focused-window KeyPress/KeyRelease tests, NOT direct StartSend. |
| A16-02 | Missing active.json, or a directory at that path, could look like a fresh empty workspace with saved revisions already present | Distinguish empty/staging-only from orphaned committed revisions; fail before old-session teardown. Never select the highest revision automatically. Require reviewed pointer repair. Test old model/client/epoch/history/private draft and released destination lease. |
| A16-03 | Preview could treat a chat-state.json directory as absent; metadata reads accepted needlessly large documents | Refuse directory/symlink state paths; pointer limit 4KiB, revision metadata 64KiB; null revision identifier is explicit invalid data. Packed load checks linked paths. Empty new workspace remains supported. |
| A16-04 | Non-final prompt tokens computed the final block's attention output and FFN even though only its K/V would be used later | Keep EVERY earlier layer and final-block Q/K/V/cache construction; elide only the unused final-block output path when computeLogits=false. Reference session option remains. Compare complete K/V and continued logits independently; add managed and TRAINED native-export regressions plus CPU prefill benchmark. |
| A16-05 | Failed CUDA discovery was permanently cached for the app lifetime, so explicit reconnect could not retry a transient failure | Cache successful probes only. Failed probe is retried on a later explicit open/reconnect. Same timeout, bounded logs and honest CPU fallback remain. Requires real failure-then-recovery target check. |
| A16-06 | Nested removal confirmation had no single-flight guard or local exception handler | Disable chooser/remove while confirming, reject duplicate events, catch displayable error and keep it after controls recover. Headless double-confirm/cancel checks. No silent permanent deletion. |
| A16-07 | Failure after disconnecting for active-model removal could reopen the model but discard the next draft/private flag | Capture and restore draft/flag after successful recovery; early refusal when trash path is a file. If reconnection itself fails, keep a visible warning rather than claim full rollback. |
| A16-08 | Programmatic control updates could leave _settingControls true after a setter/conversion exception | try/finally restores suppression; SetOnlineCheck preserves prior nesting state. Headless malformed-Ready conversion exercises release. This is not authentication of the local child. |
| A16-09 | Laptop script merged native stderr into a Stop-preference pipeline | Temporarily Continue only during invocation, log stderr, reset/check the REAL exit code, restore preference. Do not accept missing exit status or nonzero exit. Windows self-test executes the actual function with exit0/stderr and exit7/stderr. |
| A16-10 | Packager accepted app-only Linux updates while the supplied update validator supports Windows only | Refuse unsupported update target BEFORE expensive builds; explicit full Linux publication stays available. No automatic full-package fallback. |
| A16-11 | Data growth must remain secondary to this audit, while preserving evolution comparisons | Add96 training examples,16 controls,16 held-out. Freeze48 baseline texts and all14 earlier held-out files. Runtime/manifest/docs advance together. Existing workspace validation is not replaced. |

## Audited chains retained without unsupported “fixes”

Reviewed training label/position assembly, QAT/STE, optimizer update and restore, trained export, selected-target projection,
checkpoint activation, immutable stage references, online replay exclusion, selector epoch fencing, clear-chat durable boundary,
and app-only update ownership. No specific mathematical cause of the owner's incoherent text was established without their
workspace weights/state/logs. Do NOT call keyboard/layout repairs a learning-quality fix or substitute canned replies.
The trained device -> master -> quantized disk -> portable CPU check remains a mandatory CPU AND CUDA test.

## Performance semantics

The prefix shortcut removes unused computation, not history tokens, parameters, cache entries or previous layers. Prefix K/V
is still produced for every layer. Only the last prompt token and all generated continuation tokens execute the complete output
path. Reference NewSession(optimizePrefix:false) remains for tests/benchmarks. No .NET wall-clock speedup or VRAM saving is claimed.

A negative CUDA probe can now cost another bounded probe on the next explicit open. Positive caching remains; native startup
still decides actual device availability. A CPU fallback is never a CUDA pass. Normal training FP32/QAT and update cadence are unchanged.

Missing active pointer with existing revisions is an explicit recovery problem. Automatic “latest directory wins” would ignore
previous rollbacks or an interrupted activation. No weights, journals or orphan directories are silently deleted or rewritten.
File checks assume trusted local storage; no atomic defense against arbitrary concurrent path replacement is claimed.

## Executed evidence

- reports/dataset-audit-v16.json:2730 training,368 controls,272 held-out,48 basics =3418. All full sequences <=453 byte tokens.
- reports/audit16-reference.json:72 independent prefix cases over1/2/4 layers, multiple GQA ratios,1/2 planes and1/3/17/41 prompt
  tokens plus5 continuation tokens. Identical logits and K/V in those NumPy cases. Not .NET/JIT or native memory evidence.
- reports/numerical-reference.json and performance-reference-v16.json retain independent QAT, gradient, SDPA and projection checks.
- reports/static-audit-v16.json:372 managed contracts DEFINED, not executed; actual project/script structure checks.
- Wrapper evidence records three source reproduction routes and refusal fixtures in throwaway local repositories only.

## Mandatory Grok and laptop acceptance

Run every retained managed/UI/native/worker gate. Run the new actual keyboard events: Shift+Enter newline, Enter exactly one
message, repeat while held, release then another message, disabled/no-model path, busy navigation. Test Enter after selecting
text, Russian layout, IME composition, focus changes, and scale50/80/100/125% manually on Windows; no IME pass is claimed here.
Run model-dialog double-confirm/cancel, missing/directory pointers while a working model is active, and failed active-trash
recovery with private draft preservation. Clear chat must still persist a new boundary and preserve weights/queue/journal.

Run trained prefix/native parity and all previous gradient/optimizer/checkpoint tests, then prefixPreparation and all retained
benchmark sections. Repeat create/train/eval/export/reopen with real memory readings. Execute Test-LaptopChecks.ps1 on Windows;
Check-laptop calls it automatically. Inspect false-probe then successful reconnect on actual NVIDIA hardware.
Deliver only the fresh app-only Windows update by default. Libraries stay in the existing installation; no new dependency versions.
Do not waive failures, weaken assertions, count written tests as passes, or label an old ZIP as this build.

## Primary implementation references

Avalonia11.3.22 HeadlessWindowExtensions.cs (GitHub, inspected): KeyPress/KeyRelease with PhysicalKey/keySymbol.
Avalonia tunnelling events: https://v11.docs.avaloniaui.net/docs/get-started/wpf/tunnelling-events
PowerShell preference/redirected-native-stderr behavior: https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_preference_variables
Accessed2026-09-16. Those APIs do not replace actual runtime tests.
