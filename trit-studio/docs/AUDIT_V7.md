# Trit Studio: source audit 7 and staged learning experiment

## Evidence boundary

Cumulative source revision of the supplied audit6 archive. No user repository was written to.
Executed here: source/script structure, deterministic data generation, corpus integrity/isolation, independent Python numerical checks,
and synthetic local Git package reproduction. **NOT RUN: C# compilation/tests, native TorchSharp, GUI rendering, Windows or CUDA.**
The container has no .NET SDK; a direct official SDK metadata request failed DNS resolution. Grok must run every build/runtime gate.
An independent Python specification is not execution of the C# application. No speedup, fluent pretrained model or native leak-free result is claimed.

## Implemented changes

| ID | Finding / requirement | Implementation and acceptance |
|---|---|---|
| A7-01 | Automatic create/train hid the distinction between random weights, tiny pretraining and conversation training | Explicit creation modes; UI defaults to untrained. Untrained forcibly uses zero optimizer steps; baseline is separate pretrain.jsonl; conversation is explicitly selected. Legacy protocol callers retain conversation defaults. |
| A7-02 | Online learning could contaminate a before/after experiment | Fresh preferences default off. Random/basic creation leaves online off; each manual training stage finishes paused. Online learning remains available by explicit checkbox. Real worker test checks zero steps and unchanged initial weights. |
| A7-03 | Early revisions could be pruned before the owner compared them | First untrained/basic/conversation/custom stage gets an immutable inference-only copy under evolution. SHA256 and stage metadata, bounded copy buffer, no overwriting, atomic staging rename. Copies survive training-chain pruning, but are NOT resumable optimizer checkpoints. |
| A7-04 | A stage copy failure after successful training could misrepresent committed weights | Archive failure is a visible warning, not a rollback or success claim for the archive. Existing verified working checkpoint remains authoritative. No cryptographic signing/power-loss certification. |
| A7-05 | Approval of an old chat bubble recomputed context from a later, possibly pruned history | Capture the retained context at generation start; persist bounded nonrecursive learning history. Each button closes over immutable context and workspace epoch. Excluded turns form boundaries, not gaps across which unrelated history is spliced. |
| A7-06 | SiLU was assembled as separate sigmoid/multiply expressions | Use TorchSharp functional.silu (not inplace). Native/reference forward/gradient regression added, with independent Python parity executed here. |
| A7-07 | Streaming byte output repeatedly built integer lists and decode arrays; valid sampler options were repeatedly allocated | One bounded response byte buffer, direct decode of the complete UTF-8 prefix, one predicate per generation. Valid immutable sampling options return themselves. Limits, RNG path, UTF-8 and per-response state isolation retained. |
| A7-08 | Creation-stage changes could leave stale button captions; UI version labels still said audit5 | Dynamic stage/caption feedback restored after completion; title/corpus/build versions aligned. Existing-model ready chooses an appropriate next material without changing weights. |
| A7-09 | Curriculum and first-principles baseline were mixed in one starter file | 48 original baseline texts separate from 1354 conversation-training records. Existing validation is kept unchanged across stages; basic stage cannot silently include imported/conversational/learned replay targets. |
| A7-10 | Baseline-only retraining after conversation/online learning would imply a clean baseline it cannot recreate | Explicit refusal after later-stage learning; use a new model or a real checkpoint rollback. Stage names are not claims about model quality. |
| A7-11 | Corpus needed more natural low-pressure continuity and calibrated replies | +160 train, +24 control, +16 new blind challenge; all five prior blind files byte-frozen. Whole sequences fit default512. New baseline is additional, not a disguised conversation test. |

## Executed evidence

- reports/dataset-audit-v7.json: 1354 conversation train,160 control,128 held-out,48 separate baseline texts;1690 total records.
- reports/audit7-reference.json: three independent SiLU value/gradient shapes; maximum value error9.536743e-7;320 output-storage/capacity simulations; baseline data-policy checks. Not C# or GPU execution.
- reports/numerical-reference.json: retained dense Python vs incremental NumPy reference and actual toy fitting in Python. Not conversation-corpus quality.
- reports/performance-reference-v7.json: retained independent SDPA/projection/shape checks on the updated corpus, not measured C# throughput.
- reports/static-audit-v7.json:141 defined managed contracts plus structural checks; NOT a compiler.
- Enclosing evidence/package-rehearsal-v7.json: exact full/upgrade source reproduction and refusal cases in throwaway local Git repos only.

## Required actual-runtime gates

All141 managed contracts; headless stage defaults/captions plus existing feedback tests; actual worker zero->basic->conversation
transition, no hidden training with mode off, durable paused restart, baseline integrity and unchanged control corpus.
Run native SiLU parity, previous QAT/attention/projection/optimizer/cancellation tests, repeated train/evaluate/reopen soak and benchmarks.
Run Windows desktop and genuine CUDA checks on the laptop. CPU fallback is not a CUDA pass. Do not weaken tests or substitute canned replies.

## Deliberate limits

The first experiment is not a pretrained chatbot: random weights do not inherit Llama knowledge. A48-text baseline is intentionally tiny.
Conversation data are synthetic starter material, not a fluency guarantee. Repeated validation is not a blind generalization measurement.
A zero-step model still initializes storage/native training machinery; no near-instant startup claim. No models are trained by packaging.
Stage copies contain only inference weights. Continue training from the workspace master+optimizer; earlier stage copies are for comparison.
Stage-copy warnings require manual repair/re-export if desired; the first copy is never silently replaced by a later model.
Legacy chat records without captured history use only available journal context; that missing history cannot be reconstructed.
Chat history/replay and captured context remain local plaintext; exclusion is not erasure or unlearning.
FP32, expanded GQA, byte tokenizer and explicit checkpoint cleanup remain. No native performance/memory claim is made before actual measurements.

Active portable first-run instructions were updated to the actual staged defaults and corpus counts; historical audit documents remain historical. Next-stage UI defaults use the captured submitted material, not a combobox that can be updated by a later Ready event.
