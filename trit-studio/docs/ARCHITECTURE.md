# Architecture v3

`App (Avalonia)` owns chat/UI and a workspace UI lease; it spawns a single `Trainer` process over UTF-8 JSONL.
`Core` owns the model format, UTF-8 byte vocabulary, CPU inference, quantization, data format and persistence helpers.
`Trainer` owns all optimizer mutation, dataset caches, queue and checkpoint publication. It is the sole workspace writer for learning.
`Runner` uses Core only. `tools/Delivery` is a C# build/release orchestrator, not an application runtime dependency.

## Training/inference split

Master parameters are FP32. Quantized values are calculated on-device with a straight-through derivative.
The export packs ternary residual planes; the CPU reader expands them into FP32 arrays for portable vectorized inference.
This is not a custom ternary CUDA kernel, original LLamaTritLLM binary compatibility, or an imported pretrained model.
RMSNorm, causal GQA, paired RoPE, SwiGLU and tied embeddings have an independent numerical specification fixture.
Count tied weights once and include norms. Learning changes values, not architecture/parameter count.

## Consistent publication

The revision directory contains master.weights, model.tritmodel, optimizer.bin, state.json, settings.json,
base-train.json, validation.json, commit.json and revision.json (hash manifest).
Stage files, flush, rename the directory and replace active.json atomically. An abandoned staging directory is not active.
Snapshots do not embed the growing chat journal; replay commit IDs associate accepted updates with the snapshot.
For checkpoint v1 only, inputs/settings are read from its historical workspace-root layout.
Rollback changes active pointer, loads matching settings/corpus/optimizer/RNG and reconciles replay IDs, then pauses learning.

## Concurrency and interaction

Commands carry IDs and receive accepted then completed(success/error/result). Online completion means durable queue acceptance,
not completion of a gradient update; online-result and published communicate the latter.
Stop advances a cancel epoch before its queued handler, cancels the active operation and rejects older queued train/create/mode commands.
Native calls are cancellable at step boundaries, not arbitrarily interrupted inside CUDA.
The UI continues CPU inference on an immutable managed snapshot while training mutates a separate model.
Publication loads are serialized, generation/epoch fenced, and serving switches between replies.

## Resources

Variable-length cached examples, dynamic batches and length-bucketed evaluation avoid full-corpus padding.
RoPE/masks and device weights persist; transient tensors live inside disposal scopes.
Initialize and quantize with bounded Parallel.For; dense training uses LibTorch threads/device kernels.
RAM budget is an estimate and process RSS guard, not a hard allocator cap or GPU VRAM budget.
No nested unbounded parallelism or compulsory full-memory filling. An OOM produces a visible failure and last-snapshot recovery.

## Data

One record can be text, a prompt/answer pair, or alternating conversation pairs. Only the final assistant answer is supervised
for a multi-turn record; earlier pairs are context. Missing/unsupported roles fail explicitly.
Fixed token IDs do not change after online text arrives. Excluded/private exchanges never become learning context.
Validation is control data, not optimizer input; test is held out from the trainer entirely.
Approved outputs/corrections are targets, not automatically accepted model self-responses.

## Audit3 lifecycle and housekeeping

All worker disposers await the same shutdown task. Window closing cancels device discovery and waits for publication/maintenance
before releasing its UI lease. Destination UI ownership is acquired before abandoning the current workspace.
Queue discard is an acknowledged serial worker operation with an out-of-band online interrupt; it creates tombstones, not unlearning.
Maintenance stops the worker, drains asynchronous loads, retains the GUI lease and acquires the trainer lease. It preserves the active
ancestry up to eight snapshots, not an unrelated newer branch. No live deletion or automatic deletion occurs.
`revision-counter.json` is a high-water mark outside rollbacks. Reserve before publication and preserve before pruning; gaps are valid.
ChatJournal reads a bounded tail and isolates partial last records before append. History remains local plaintext.
New bundled seed data is optionally merged only on manual fine-tuning; checkpoint-local validation is not silently replaced.
