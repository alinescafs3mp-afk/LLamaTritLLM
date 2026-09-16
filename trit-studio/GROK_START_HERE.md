# Grok: audit16 deep regression pass, app-only Windows delivery

Read enclosing 00_GROK_START_HERE.md first. Target owner fork alinescafs3mp-afk/LLamaTritLLM, existing master.
Actual verified fork baseline is fa9c9261fdcf3f463855ab5fb2687e71ba4654d9 (integrated14). Exact15 upgrade is also provided.
Preserve mean(long[]), protocol IOException and Linux EOF host repairs. Never reset/rebase/force-push to erase owner work.
The author did not compile/run C#, Avalonia, PowerShell or CUDA. Do not convert Python/static evidence into runtime passes.

From trit-studio run bash scripts/deliver.sh (or scripts/deliver.ps1). Default MUST be artifacts/TritStudio-win-x64-update.zip,
containing only owned binaries/metadata/data plus both trainer programs. No vendor CUDA/LibTorch/Avalonia/.NET redistribution.
Do not invoke --full to evade compatibility checks. Keep pinned package versions; any forced change needs owner approval and
an explicit migration path for retained libraries. Actual Linux-to-Windows publishing is not a Windows runtime test.

Read docs/AUDIT_V16.md, PERFORMANCE_V16.md, UPDATE_RU.md and QUALITY_DIAGNOSTICS_RU.md. Run all372 managed contracts, all
headless/real-child/native/worker tests, new Audit16UiChecks and every benchmark. Fix causes and rerun without weakened tests.
New keyboard tests MUST use focused-window KeyPress/KeyRelease, not StartSend invocation. Verify Shift+Enter, held Enter,
release/next send, no model and busy-gate refusal. Raw event behavior/IME and DPI also need real Windows desktop acceptance.

Check missing active pointer with revisions, pointer/state directories, linked metadata, null/oversized pointer metadata,
failed opening preserves active model/private draft/history/epoch and releases its candidate lease. Do NOT guess the latest
revision or create new random weights over an orphaned workspace. Test nested delete dialog duplicates and failure feedback.
If active removal fails and recovery succeeds, retain its draft/privacy. A failed native recovery must remain visible.

Run trained native -> packed disk -> CPU parity, including new prefix shortcut vs previous compute path and continuation KV.
Actual prefixPreparation benchmark is CPU only. No promised speedup threshold. Repeated train/eval/reopen memory soak remains.
Run packaging/Test-LaptopChecks.ps1 on Windows; Check-laptop calls it. Probe CUDA failure then explicitly reconnect after recovery.
A negative probe may be retried; a CPU fallback or nvidia-smi output is not a successful CUDA training test.

3418 corpus records:2730 train+368 controls+272 held-out+48 basics. All14 previous holdouts/baseline frozen. No pretrained weights
or conversational-quality guarantee. Existing-model controls never change silently. Staged zero/basic/conversation experiment,
clear-chat boundary and exclusion privacy behavior remain. Diagnose actual owner weights/logs before explaining gibberish.

Return absolute fresh UPDATE zip path, SHA256, output directory, WHERE_TO_PICK_UP.txt, full logs, actual host results and final
fork commit. Explicitly list pending Windows/CUDA tests. Do not return an old ZIP as this build or a giant portable archive.
