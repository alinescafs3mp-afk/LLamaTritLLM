# Third-party notices

Trit Studio's residual multi-plane quantization and Llama-style implementation are based on study/adaptation of **LLamaTritLLM** by virex-84, MIT licensed.

- Repository: https://github.com/virex-84/LLamaTritLLM
- Reviewed commit: `82e0b751cc7b81b0b73a522e141de6ff45e40a71` (2026-09-11).
- Original license is preserved at `third_party/LLamaTritLLM/LICENSE.txt`.
- The principal adapted mechanism is `training/core/MultiPlaneTritPacker.cs::FromFloat`: mean-absolute residual scales, multiple ternary planes, and compact base-3 packing.
- This is not an endorsed upstream release. The byte tokenizer, GUI, Torch trainer, worker protocol, persistence, and file format are new. Original `.bin` checkpoints are not compatible with `.tritmodel`.

Runtime package dependencies are restored from NuGet, not copied into this source archive:

| Package family | Pinned version | Role |
|---|---|---|
| Avalonia.Desktop / Avalonia.Themes.Fluent | 11.3.22 | Desktop GUI |
| TorchSharp-cpu | 0.107.0 | Default trainer and LibTorch CPU bundle |
| TorchSharp-cuda-linux / TorchSharp-cuda-windows | 0.107.0 | Optional trainer build variants |

Preserve package licenses/notices when distributing compiled builds. Dependency licenses are not replaced by this repository's license. No font files, pretrained weights, or native binaries are distributed in the source archive.
