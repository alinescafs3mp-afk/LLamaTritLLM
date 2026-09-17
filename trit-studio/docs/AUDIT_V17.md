# Trit Studio audit17: investigate learning first, phase-correct memory and three-block UI

## Evidence boundary and integration baseline

Owner fork: `alinescafs3mp-afk/LLamaTritLLM`, `master`, commit `1570538ee47af73a1c30d38acaf91e39a82714e5`.
The exact deployed trit-studio Git tree was reproduced locally as `f3d4676ff7ac4ca8a2dd1968a09d6fed48b44abb`.
Retain ALL Grok host fixes: mean(long[]), protocol IOException, Linux duplicated stdout handles, and AppendAsync in Audit15UiChecks.
No owner checkout or remote was changed. This is a cumulative SOURCE handoff; compiled delivery is Grok's job.

Executed here: independent real Python gradient training experiments, a trained V1 fixture with separate NumPy disk inference,
old neural reference checks, corpus/hash checks, static/source checks and local Git package rehearsals.
**NOT_RUN: .NET compilation, C# tests, TorchSharp native execution, Avalonia layout/events, PowerShell, Windows and CUDA.**
The container has no dotnet/C# compiler. A Python result is NOT a native pass. Source code counts are not runtime evidence.

## Findings and implemented repairs

| ID | Finding | Change and mandatory native acceptance |
|---|---|---|
| A17-01 | Screenshot has 8000 steps but LR1e-6 and byte-control loss3.7847. Prior gates proved numerical transport and loss reduction, not that the complete application could learn even six replies. | Add `LearningSmoke` to the REAL trainer self-test: fresh 115392-param model, real gradients, six short Russian pairs, <=1000 updates at.001, export/reload with production V1/CPU generator, require all6 exact and teacher loss<.20. Run CPU and CUDA. No lookup replies or relaxed assertion. |
| A17-02 | No concrete way to separate weak learning from corrupted export/inference for the owner's actual checkpoint | Explicit `quality` command: verified committed file, live training step match, small training/control sample losses and teacher-forced byte accuracy, 3 fixed greedy replies, native/packed CPU logit comparison including optimized prefix. Persist local diagnostic JSON; no training/queue/weights changes on successful probe. Actual worker invariant tests. |
| A17-03 | Very small LR was numerically valid but easy to miss among tiny controls | Scientific+decimal warning and explicit confirmation below1e-5. One-click profile sets next LR.001,B8,length<=512,2000steps,publish200. Does not start training or alter architecture. Independent Python contrasts support the warning, not a unique diagnosis of owner's weights. |
| A17-04 | Creation/loading validated a full B32*T1024 training envelope against the configured RAM quota, even at zero steps | Split configuration, initialization and actual-training preflight. Creation only checks persistent envelope. CPU training validates actual dynamic batch length. CUDA activations are not compared to a host RAM quota. Existing host RSS and native allocation error recovery stay. |
| A17-05 | New-model creation reused the SELECTED model's batch, sequence, LR, files and RAM fields | Separate creation controls/resources/training plan, unused training controls hidden in zero-step mode. New model only consumes its own built-in stage. Custom files belong to selected-model fine-tune. UI/worker regressions cover independent options. |
| A17-06 | Six scattered cards and misaligned header obscured the target model | Exactly3 top-level training cards: current model, create another, fine-tune selected. Current info full-width, creation/fine-tune side by side or stacked when narrow. All header controls centered on the same row; caption separate. Preserve selected identity and navigation guards. |
| A17-07 | Pure initial-tensor or same-implementation checks cannot independently establish deployed decoder behavior | Ship one clearly labeled80,590-byte synthetic learned fixture under tests. Separate Python trainer, V1 encoder, NumPy binary reader/autoregressive inference verified all6 exact. Managed test must load it and generate all6 via actual C#; no special-case inference logic. Never selected/injected into new user models. |
| A17-08 | Diagnostic result could remain visible after switching models | Clear model-specific quality summary at successful model switch. Preserve failed-switch state and all prior draft/privacy protections. |

## What was and was not diagnosed

Old formula exactly reproduces **28342MiB** for the experimental4788992-parameter model, batch32, length1024:
694.445MiB persistent envelope +3072MiB linear term +24576MiB attention term. It compares against configured16384MiB,
not measured free system memory or the physical16GiB GPU. FP32 parameter bytes alone are18.269MiB.
The formula is conservative. Even the persistent estimate is NOT measured RSS/VRAM, reservation or a peak guarantee.
CUDA activation failures still need actual hardware handling; no free-VRAM probe/pinned memory/mixed-precision promise is added.

LR1e-6 is1000x below the application's original1e-3 default. Independent fixed-six-example training: after8000 steps at1e-6,
0/6 exact replies; at3e-4 all6 by300; at1e-3 all6 by100 and still6 at300. These runs use analogous math but DIFFERENT initialization
and sampling from C#. They do not prove the saved LR used by the owner's actual weights, nor make the main corpus a fluent chatbot.
The diagnostic report records the saved manual LR rather than relying on whatever the next-run editor currently displays.
Teacher-forced loss/accuracy is not free-running generation or a generalization score. Random-byte generation can form valid
Unicode from many alphabets: a screenshot of odd glyphs does not by itself prove a text-encoding failure.

No decoder language filter, canned responses, silent dataset injection into raw creation or hidden architecture change was added.
The supplied six-reply fixture is an explicit TEST model, not stock pretraining, a user-workspace migration, or a universal assistant.

## Dataset and files

v17 adds32 reviewed training,8 controls,8 held-out:2762 train,376 controls,280 held-out,48 frozen basics =3466 records.
All previous held-out bytes and48 baseline texts remain unchanged. Fixed262-ID byte vocabulary and TritStudio V1 format unchanged.
The supplemental six-reply TEST MODEL is separate from the corpus and never added to ordinary model creation.
No vendor library versions were changed. Default Windows delivery remains the small app-only update; the explicit test fixture
adds~79KiB at checks/learning-reference.tritmodel, not CUDA libraries or someone else's large pretrained checkpoint.

## Required execution and limits

Run all existing gates plus Audit17Checks, Audit17UiChecks, actual large zero-step/reopen and read-only-quality worker fixtures,
native LearningSmoke on CPU and actual CUDA, and the C# independent learned-fixture recall. The build MUST stop if any fails.
Keep full logs with each generated reply and real exit codes. A declining loss alone no longer passes the learning gate.
Run Windows3-card/header layout checks at100/150/200%DPI and all supported UI scales. No UI render pass is claimed here.
Quality JSON contains only fixed public prompts/generated replies and aggregate metrics, not copied raw user records;
GENERATED replies can contain learned personal information. Inspect before sharing. Diagnostics don't erase/unlearn anything.
All prior small-update compatibility checks, clear-chat boundaries, stage references and read-only inference remain.
