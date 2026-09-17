# Trit Studio audit22: paraphrase transfer, final-fact priority and actual batch balance

## Baseline and evidence boundary

This cumulative starts from the exact last delivered v21 archive with388context training rows and16paired tests.
Owner fork master was read:68727516a4cb06bf99e971991bd95d9dd6632d4c, subtree1c31d80d6dd478091f19d6fa2cc6ff58f864bc5e.
No owner checkout or remote was changed. Preserve all integrated host repairs. The author executed Python data/algorithm
checks, actual independent learning and synthetic local Git package rehearsals. **C# build/tests, TorchSharp native runtime,
Avalonia, PowerShell, Windows and CUDA were NOT RUN here.** No dotnet/compiler is installed. Python is not native acceptance.
BinaryV1,262UTF8-byte IDs, vendor versions and small-update policy are unchanged. Previous reports remain historical.

## Findings and implemented source changes

| ID | Finding | Repair and required native regression |
|---|---|---|
| A22-01 | v21expanded context priority included408intermediate targets among796views; many are acknowledgements rather than retrieval of a final fact | Prioritize only original final fact targets. Keep all intermediate supervision in general corpus. This is a distribution finding and hypothesis about generic answers, not a unique causal diagnosis. |
| A22-02 | Reference-pool proportions did not guarantee actual batch proportions because length buckets could isolate homogeneous examples | Opt-in DialogueBatchPlanner chooses category per slot. No subsequent bucket resampling. Stable phase patterns, rotating batch1..64, exact targets/length and session-owned RNG. Native batch8/64 and optimizer-resume checks. |
| A22-03 | Large grids of one final-question form could dominate the final-fact pool | Choose final-question group then row uniformly; original targets only. This balances literal forms, not semantic intentions, and changes selection probabilities explicitly. |
| A22-04 | A few fixed question forms encouraged reproduction instead of transfer | Add160short intent paraphrases plus1278crossed intro/question/fact exercises, including corrections and absent information. All old sources/control/heldout/basic remain byte-identical. This is authored template data, not broad language pretraining. |
| A22-05 | Existing known recall and16pairs did not cover ordinary new conversation formulations | Retain old reports and add62case GeneralizationProbe with own generated history, exact reference plus all text, recorded-training exclusion and visible per-case progress. Full manual62 vs automatic8 labelled. No evaluation target enters decoder/training. |
| A22-06 | Adding a recipe can reintroduce stale settings and silent opt-in | TransferPractice defaultsfalse, requires Course+ContextPractice, independent creation/selected drafts, explicit profile enables, parent opt-out clears. Test save/Ready/reopen/report identity. |
| A22-07 | New data/report paths can omit adjacent update files or mutate successful training state when report fails | New files in manifest/allowlists/all own update copies; only diagnostic file uses read-only loader. Report failures warn after commit; successful probe changes no weights/replay/mode/active pointer. Actual worker invariance gate. |
| A22-08 | Building the unused legacy3phase reference pools would duplicate preparation for the new sampler | Construct legacy ConversationCurriculum only outside TransferPractice; new phase names preserve bounded report checkpoints. No token-array duplication by the new index planner. |
| A22-09 | A different reference acknowledgement could hide that the actual generated input was already trained | Check reference AND actual generated histories. Include saved sources plus up to256recent learned replay entries; older/deleted data remain unknown. Add real deterministic-generation exclusion regression. |

A drafting audit also caught18wrong-gender name questions and one development case ("мой имя"). They were corrected
before release, a regression was added, and BOTH full independent A/B runs were repeated on the corrected final corpus.
Preliminary raw reports and their exact data-only correction patch remain marked as preliminary, not current evidence.

## Real semantics

For batch8, phase0general/language/facts=2/4/2;phase1=2/3/3;phase2=3/3/2. Batch16doubles and64multiplies by8.
For small/odd batches category positions rotate by completed*batchSize modulo8; batch1is not permanently stuck on language.
The general component can itself contain facts and short replies. These are sampling-source quotas, not mutually exclusive
semantic labels. Every example has nonzero general probability in every phase, not guaranteed coverage per step.

No bucketing on the already chosen mix: padding/cost can increase. The old route is unchanged when the flag is false.
Index arrays refer to owned immutable encoded records. In-place mutation outside the owner remains unsupported.
Final fact questions are grouped literally; two synonymous questions remain different groups. Conditional answer balancing
within a group is not generally guaranteed. The fixed training objective/gradient clipping/native math remain unchanged.

RNG is the actual training session's saved generator. A cancelled partial selection may advance it; worker recovery restores
last committed state. Pre-cancel consumes nothing. New requested jobs restart the phase/LR schedule on existing weights.
No infinite learning or automatic snapshot deletion was added. Validation regression guard and all storage checks remain.

Diagnostic results are PUBLIC development evidence, not a hidden benchmark or general dialogue score. New phrasings are
reserved within the authored intent/template families; vocabulary/topics deliberately overlap. Exact matching can reject
acceptable paraphrases; read all outputs. Both old16pairs and open tasks are retained to expose regressions.
Recorded-training exclusion covers current saved source input/history plus256recent learned examples, for both reference and generated contexts; not unknown older/deleted/historical/private data. Generated
text can contain learned private information. Reports are not imported into training. Manual check can take time, is cancellable,
and leaves committed state unchanged; report-file failure may leave one companion file without the other.

## Data/independent evidence

5955physical source rows:5097course train,408control,402heldout,48separate basic. Independent all-assistant expansion gives7851views.
New train:160intent paraphrases and1278fact templates;new public tests62. All whole sequences fit512,max453byte input positions.
All previous data files except the updated manifest are byte-identical. Check full/effective control and derived probe inputs.

reports/audit22-reference.json:1664quota cases,700deterministic sampling/target-count comparisons, preserved original-reference
and priority counts, full+derived diagnostic/control isolation, expected-response length bound. These are Python specifications,
not C# execution. Retained numerical_reference and performance_reference validate analogous neural formulas, not runtime lifetimes.

reports/transfer-legacy-v22.json and transfer-balanced-v22.json are the completed independent same-data/seed6000step A/B trials.
See LEARNING_RESULTS_V22_RU.md for full results, limitations and bad responses. They use821120params,B8,CPU,not user's4.79M native
model. Holdout generations were evaluated at the end only. No train target filtering based on successful outputs.

## Required real acceptance

Run all retained managed/UI/worker/native gates, including Audit22Checks, Audit22UiChecks and DialogueBatchSelfTest.
Require actual direct-vs-staged training with new recipe and read-only full62case report state/hash invariance. Keep all old
learning/export/optimizer/independent six-reply fixture gates. Do not replace responses with strings or weaken checks.

Run an isolated native --learning-lab --course --context-practice --transfer-practice --enforce-guard experiment, then inspect
known/new/open outputs separately. CPU pass is not CUDA. Exit3means rejected candidate, not full completion. Short pilots must
be declared; full target CUDA training remains a separate requirement. No claimed speech quality from build/test counts alone.

Default delivered binary ZIP: TritStudio-win-x64-update.zip, both own trainers/data, NO vendor libraries. Check all adjacent
corpus versions after merging into the existing installation. No dependency upgrade, full-package fallback or user-model overwrite.
