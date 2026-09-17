# Trit Studio audit20: conversation-oriented learning, not a larger accuracy badge

## Baseline and evidence boundary

Cumulative source on the supplied complete audit19 source, retaining the owner's deployed audit18 changes.
Verified owner master:68727516a4cb06bf99e971991bd95d9dd6632d4c; exact trit-studio tree reconstructed from the
v19 reverse patch:1c31d80d6dd478091f19d6fa2cc6ff58f864bc5e. No user checkout or remote was changed.

Executed: independent real Python learning, data generation/hash/split checks, numerical objective checks,
source/project/script checks and throwaway local Git package application. **NOT_RUN: C# build/tests,
TorchSharp runtime, Avalonia/Windows, PowerShell and CUDA.** dotnet is absent and an official SDK metadata
download failed. Independent Python models have different initialization and sampling, and do not validate
native lifetimes or actual UI event delivery. Previous reports are historical. New actual gates must be run.

## Findings and implemented changes

| ID | Finding | Source repair and required actual acceptance |
|---|---|---|
| A20-01 | A multi-turn training record taught only its LAST assistant response; previous correct answers were input context only | Explicit course expands each prior assistant turn into its own masked target with preceding real context. Original rows/IDs stay. Derived full/effective input keys are rechecked against frozen controls; reject original contamination and omit derived control collisions with a count. Stable IDs/dedup, cancellation and50000 source-view limit. Legacy opt-out is unchanged. |
| A20-02 | Starting with every long/context-heavy task at once did not establish reliable short replies | Opt-in three-pool course:20% short starter,40% dialogues+starter replay,40% full union+starter replay. Same weights/optimizer; immutable reference pools. Explicit6000step/B16/512/LR.001 scheduled profile, not auto-start. Full union is saved, never pretend phase rows are more unique data. |
| A20-03 | Mean over ALL target bytes gives long answers more objective weight than short examples | Optional EqualExampleWeight uses unreduced CE and weights1/(batchRows*rowTargetCount). Each row sums to equal importance; attention/context stay complete. Token-mean full-control CE and existing byte-accuracy denominator stay unchanged. Native per-row CE/all-gradient oracle, selected/full, unequal targets and actual updates required. |
| A20-04 | High teacher accuracy and three independent questions do not test free-running conversational context | Public10-case ConversationProbe uses committed packed model and OWN generated history. Four open tasks need manual review; six simple checks are lexical/exact heuristics, not semantic/hidden benchmark. Dropped history or looping cannot pass an automatic case. No expected answer in prompt or generation. |
| A20-05 | Owner needed inspectable evolution rather than another successful optimizer completion | Explicit Check conversation button; JSON+readable Markdown, model revision/step/hash, actual text and dropped-history counts. During course, bounded phase-change/first/final checkpoint probes; reporting failure warns after successful commit, never rewrites saved weights or retries on every checkpoint. No probe training, no persistent chat/replay mutation. |
| A20-06 | Training switches could regress into prior settings-loss/target-model bugs | Course/loss flags in per-workspace next-run and separate creator drafts, UI hydration/autosave, explicit profile vs Save; stale report epoch rejected. Legacy JSON defaults false. Raw creation still zero steps and original48 basic texts only. |
| A20-07 | A new data filename could be omitted from app-only trainer updates or mistaken for a control set | conversation-starter.jsonl is an explicit training file in BundledCorpus and owned payload for both trainers/checks. challenge-v20 remains test-only. Loader version/hash/count checks, manifest-driven tests and old split rejection stay. |
| A20-08 | Small recipe tests did not expose the remaining generalization gap | Real independent foundation and full-course CPU experiments record known/novel text, not only percentages. An actual native full-course lab is included and required for quality evidence. The foundation experiment reached exact known recall but NOT robust new-question conversation. No fluency claim. |

## Important semantics

This is an opt-in training recipe, NOT a new pretrained model. No weight download, source-language filter,
response lookup, output substitution, automatic expected-answer injection or vendor upgrade was added.
Tokenization remains262 fixed UTF8-byte IDs; binary formatV1 remains. Existing weights are not migrated.

The legacy48-text basic stage is unchanged and byte-frozen. conversation-starter is a DIFFERENT optional
train file. Selecting a course explicitly authorizes adding it and derived assistant views. A new manual
course begins the three-phase schedule again, including on a positive-step model. For plain continuation,
turn Course off; the already saved expanded data remain and EqualExampleWeight can stay explicitly on.
Changing this training objective changes what displayed training loss means; control loss remains comparable
only within its unchanged data/budget. Live byte accuracy can go down and does not measure answer quality.

Prefix expansion adds context computation; equal-example CE adds a weight vector and unreduced loss.
This release is NOT a measured throughput optimization. Replaying short rows intentionally changes sampling
probabilities. Frozen control candidates are still checked; no weakened regression gate to complete a run.
Existing finite snapshot capacity, optional interval planner, manual cleanup and warmup/cosine semantics stay.

ConversationProbe is PUBLIC DEVELOPMENT evidence. Exact source forms of these prompts are not automatically
added to training, but overlap in subject/meaning and prior data can exist. A correct keyword alone can be
misleading; a paraphrase may be scored false. Read complete outputs. Public probes are not a blind score,
not a generalization guarantee and not a replacement for untouched test data. Four open-ended tasks are
explicitly excluded from the automatic denominator. Empty/poor open answers never become automatic passes.

Within a case, history contains the model's actual output, not reference replies. Separate cases start empty.
Reports include generated text, which can contain learned personal information; inspect before sharing.
Manual successful checks do not change weights/optimizer/active pointer/mode/replay. They write only a new
local diagnostics report. Automatic checks read a committed model and run between training intervals.
A cancelled check may leave a JSON report without its Markdown companion; weights remain committed.

## Evidence and runtime acceptance

reports/audit20-reference.json independently checks16 objective cases with gradients,8 curriculum boundaries,
all actual derived dataset targets/masks and isolation. It finds3247 original course rows and1320 extra
assistant targets =4567 encoded views before replay, NOT4567 independent authored conversations.
Dataset:2966 main +281 starter +408 controls +308 heldout +48 basic =4011 records. All full sequences<=453.

reports/conversation-foundation-v20.json: independent medium821120 params,B8,1600steps,schedule/equal loss,
281starter only. It ends with7/7 exact KNOWN probes; two of three new formulations remain poor. This is
memorization/limited transfer evidence, not a conversational product acceptance. All raw outputs are retained.
reports/conversation-full-v20.json is the additional full-course experiment; check its last observation and
actual text before interpreting. It does not implement the worker's checkpoint regression/retention policy.

Run Audit20Checks, Audit20UiChecks, ConversationObjectiveSelfTest, course direct-vs-staged real worker parity
and read-only report invariants, alongside ALL older managed/UI/native/cancellation/optimizer/fixture gates.
Then run the native --learning-lab --course on an isolated directory and report real free answers, including
failures. CPU completion is not CUDA execution. Neither a lower loss nor a passed six-reply smoke is fluency.

Cumulative packaging proves exact source application and refusal cases, NOT compilation or runtime behavior.
Default delivery is the small win-x64 update, both trainers/data, no vendor files. Baseline owner work is retained.

## Primary references inspected

- TorchSharp src/TorchSharp/NN/Losses.cs at8f4def03b641b6753f18076aa5438f8eaaef2d30: cross_entropy supports Reduction.None.
- https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html : unreduced CE and reduction semantics.
- https://arxiv.org/abs/2305.07759 : TinyStories is motivation for controlled simple corpora, NOT evidence our Russian corpus works.
Do not conflate those external results with this application's experiments.

## Completed independent full-course result

The additional independent medium821120-param CPU/B8 run completed6000steps, not the4.79M C# default.
At6000:full control loss 0.977472, teacher byte accuracy 67.5722%.
Known greeting/book prompts are correct. New phrasing and multi-turn name/correction cases still fail
(the final name answer is «Можно помогалить.», correction answer «На полке.»). This is NOT acceptable
general conversation. The full raw outputs and all intermediate observations are retained, not selected
only for attractive examples. The smaller model, independent initialization/sampling and lack of worker
regression-stop enforcement prevent a direct comparison to the owner's actual model.

The recipe addresses missing supervision and measurement; it is a tested development direction, NOT a
claim that increasing the teacher score alone solved language/generalization. Native larger-model
course evidence and richer language data remain necessary for the owner's conversational goal.
