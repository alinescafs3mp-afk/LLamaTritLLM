# LLamaTritLLM — Ternary (BitNet-1.58-style) Llama in pure C#

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)

**LLamaTritLLM** is a from-scratch C# implementation of a **BitNet 1.58-bit style**
LLM running on the **Llama architecture** — trained **on CPU only**, with no
external ML frameworks, no GPU, no CUDA, no Python.

Every **trainable parameter** (embedding table, Q/K/V/Out attention
projections, FFN gate/up/down, LM-head) is stored as **multi-plane ternary
values** `{-1, 0, +1}` with per-group scales. The result is an LLM whose
**entire trained model file is measured in *kilobytes*** — not gigabytes.

> **Activations are not stored / not quantized by default.** Intermediate
> activations (embedded tokens, hidden states, attention/FFN outputs) are
> computed at **runtime as plain FP32** and never saved into the model file.
> RMSNorm gamma vectors are also FP32. Only the *weights* are ternary —
> that's what makes the file tiny.

> ⚠️ This is an educational / experimental project (built for a research
> article). It is not a production-grade inference engine.

---

## 🌍 The headline: an LLM in ~28 KB

A full generative language model — tokenizer, weights, everything you need to
chat — fits in a **~29 KB binary file**. Even the most extreme configuration
drops below **18 KB**.

| Benchmark (same training corpus, Russian texts) | Minimal | Extreme |
|---|---|---|
| Embedding dim | 64 | 32 |
| FFN hidden dim | 64 | 32 |
| Heads | 1 | 1 |
| KV heads | 1 | 1 |
| Layers | 1 | 1 |
| Max seq len | 512 | 512 |
| BPE vocab size | 16000 | 16000 |
| Trit planes | 1 | 1 |
| **Parameters (latent)** | **106 496** | **49 152** |
| Training loss (reached) | **0.26** | **0.42** |
| **Master model file (FP32 weights + tokenizer)** | **420 KB** | **196 KB** |
| **Ternary model file (packed weights + tokenizer)** | **28.9 KB** | **17.1 KB** |
| └ *of which tokenizer data* | 3.71 KB | 3.71 KB |

In other words, the *actual network weights* of the ternary model are only
**~25 KB** (minimal config) or **~13 KB** (extreme config) — the tokenizer
takes a quarter of the file.

A model this small is almost entirely about the **ternary quantization format**:
printed on paper, even the "extreme" model's weights barely fill a page.

---

## How the size is achieved

### Two models, one training pipeline

| | Master model (`TernaryTraining`) | Ternary model (`TernaryInference`) |
|---|---|---|
| Role | trains on CPU | runs inference only |
| Weights | FP32 (latent) | packed ternary `{-1,0,+1}` |
| File | `master_model.bin` (large) | `ternary_model.bin` (tiny) |
| Purpose | training + export | deployment / chat |

1. You **train** the master model with quantization-aware training (QAT):
   on every forward pass the FP32 weights are quantized to ternary on the fly
   (straight-through estimator), so the model learns weights that survive
   aggressive quantization.
2. You **export** (`SyncToInferenceModel`) → the ternary model packs the
   weights into a compact bit-format and writes `ternary_model.bin`.
3. You **ship** only the ternary model — it can be loaded and chatted with by
   the small inference program, which contains **zero training code**.

### What is on disk (ternary format)

The model file stores **only trained parameters** — no activations, no
computed states. Each weight is a sum of `planes` ternary layers:

```
w ≈ Σₚ scale[p] · t[p],   where t[p] ∈ {−1, 0, +1}
```

- **Trits** are packed **five per byte** in a base-3 encoding (each trit is
  `log₂(3) ≈ 1.585 bit`, so 5 trits ≈ 7.9 bits ≈ one byte) — never stored as
  raw bytes or floats.
- **Scales** are stored as **FP16** (one scale per group of `GroupSize`
  weights, `ScaleMode.PerGroup`).
- **RMSNorm gamma** vectors (`norm1`, `norm2`, `final_norm`) are few and tiny —
  FP32, a negligible fraction of the file.
- `planes = 1` → classic **1.58-bit** (BitNet); `planes = 2` → INT3-like;
  `planes = 6` → INT6-like. More planes → more precision, proportionally
  larger file.

Result: ~13–25 KB of weights for a fully functional conversational LLM.

---

## Llama architecture (implemented from scratch)

- **RMSNorm** before attention and FFN (with per-block gamma)
- **Multi-head attention with GQA** (grouped-query attention) + **RoPE**
  positional encoding (`theta = 10000`)
- **SwiGLU FFN** (gate × up → down)
- Pre-norm **residual blocks** with **tied token embeddings** (LM-head shares
  the embedding matrix)
- KV-cache during inference, causal mask during training
- Optional experimental `AttentionMode`: `Softmax` (default) / `TopK` / `Hardmax`

Everything — autograd engine, AdamW optimizer, tensor ops, RoPE, BPE
tokenizer — is hand-written C#.

---

## Repository layout

```
├── LLamaTritLLM.sln                  # Solution with both programs
├── training/                         # ═══ Program "TernaryTraining" ═══
│   ├── TernaryTraining.csproj        #   builds the full training app
│   ├── Program.cs                    #   CLI: create / train / tune / export / chat
│   ├── core/                         #   shared: config, packer, tokenizer, serialization
│   ├── ternary/                      #   ternary inference model (model, blocks, KV-cache)
│   └── train/                        #   master model, autograd, AdamW, trainer
└── inference/                        # ═══ Program "TernaryInference" ═══
    ├── TernaryInference.csproj       #   builds ONLY core + ternary (no training code)
    └── Program.cs                    #   load ternary_model.bin and chat
```

The `inference` program compiles from just `core/` + `ternary/` — the training
stack (`train/`) is **never** linked into the deployment binary.

---

## Requirements

- **Windows 11** (primary), .NET SDK **8.0+**
- A CPU is enough — training is deliberately CPU-only

---

## Build

```bat
dotnet build LLamaTritLLM.sln
```

or build each program separately:

```bat
cd training
dotnet build        :: produces TernaryTraining.exe

cd ..\inference
dotnet build        :: produces TernaryInference.exe
```

---

## Usage

### 1. Training program — `TernaryTraining.exe`

```bat
cd training\bin\Debug\net8.0
TernaryTraining.exe
```

Interactive menu:

| Menu | Action |
|---|---|
| **Create model** | choose dims → BPE tokenizer trains on your corpus first → model created |
| **1. Pretrain** | language-model pre-training on `data/pretraining_data.json` |
| **2. Tune** | chat fine-tuning on `data/chat_training_data.json` |
| **3. Save** | save master weights (`models/master_model.bin`) |
| **5. Export ternary** | pack master → `models/ternary_model.bin` |
| **8. Interactive chat** | chat using the ternary model |
| **9. Exit** | — |

Typical flow: **create model → pretrain → tune → save → restart → load →
export → chat**. Defaults mirror the benchmark configs above
(`emb=64, hidden=64, heads=1, kv=1, layers=1, seq=512, vocab=16000, planes=1`).

Data files (`data/*.json`) are expected next to the executable
(`training\bin\Debug\net8.0\`).

### 2. Inference program — `TernaryInference.exe`

```bat
cd inference\bin\Debug\net8.0
TernaryInference.exe models\ternary_model.bin
```

Command line:

| Arg | Meaning |
|---|---|
| `ternary_model.bin` (positional) | path to the ternary model file |
| `--temp 0.7` | sampling temperature (default 0.7) |
| `--prompt "text"` | run a single prompt and exit |
| `--help` | usage |

Without `--prompt` it starts an interactive chat:
`exit` quits, `temp N` changes temperature.

**One honest note:** a 17–29 KB network is a *very* slim brain. Don't expect
conversations deeper than the (deliberately tiny) training corpus. That's the
point of the project — to push the *lower bound* of what an LLM can be.

---

## Model file formats

| File | Magic | Contents |
|---|---|---|
| `master_model.bin` | `TRAINABLE_LLAMA_WEIGHTS_V1` | config + BPE tokenizer + FP32 tensor dumps |
| `ternary_model.bin` | `TERNARY_LLAMA_V1` | config + tokenizer + packed ternary layers + FP16 scales |
| packed layers | `MPTP_V2` | FP16 scales + base-3 packed trits (5 trits/byte) |
| tokenizer | `BPE_COMPACT_V1` | special tokens + merge pairs as varints |

---

## License

[MIT](LICENSE.txt) — Copyright (c) 2026 virex-84