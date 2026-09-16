# Audit13 references

Official API contracts consulted2026-09-16:
- https://learn.microsoft.com/en-us/dotnet/standard/simd
- https://learn.microsoft.com/en-us/dotnet/api/system.numerics.vector-1?view=net-10.0
- https://learn.microsoft.com/en-us/dotnet/api/system.io.stream.read?view=net-10.0
These describe APIs, not evidence of executing this application. No current owner checkout/remote was mutated.

# Reviewed primary sources

Reviewed for this cumulative on 2026-09-16. Online package versions may change; package references are pinned, not "latest".

- Upstream source and MIT license: https://github.com/virex-84/LLamaTritLLM/tree/82e0b751cc7b81b0b73a522e141de6ff45e40a71
- TorchSharp project and memory/API docs: https://github.com/dotnet/TorchSharp
- TorchSharp CPU 0.107.0 package metadata: https://www.nuget.org/packages/TorchSharp-cpu/0.107.0
- TorchSharp Windows CUDA 0.107.0 metadata (LibTorch 2.10.0 / CUDA 12.8): https://www.nuget.org/packages/TorchSharp-cuda-windows/0.107.0
- TorchSharp Linux CUDA package family: https://www.nuget.org/packages/TorchSharp-cuda-linux
- Avalonia package family/current 11.3 maintenance line: https://www.nuget.org/packages/Avalonia
- Avalonia Linux runtime dependencies: https://docs.avaloniaui.net/docs/deployment/linux
- .NET installation on Ubuntu, including 24.04 and 26.04: https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu-install
- .NET installation scripts: https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-install-script

The TorchSharp README mentions older CUDA 12.1 bundles in some sections. For this version, use the selected NuGet package metadata and actual dependency graph rather than blindly copying an old README example. The Linux CUDA package must be restored and hardware-tested by Grok; it was not executed in the authoring environment.

## Audit2 reference verification (2026-09-16)

- TorchSharp optimizer state/device APIs: https://github.com/dotnet/TorchSharp/blob/main/src/TorchSharp/Optimizers/Optimizer.cs
- Avalonia manual headless sessions: https://docs.avaloniaui.net/docs/testing/setting-up-the-headless-platform
- Fork metadata and default branch verified through GitHub connector: alinescafs3mp-afk/LLamaTritLLM, master, 82e0b751cc7b81b0b73a522e141de6ff45e40a71.
- Package pinning is not proof of compilation; actual NuGet restore/build gates remain mandatory.

## Audit4 inspected APIs

- TorchSharp SDPA wrapper, including positional causal argument: https://github.com/dotnet/TorchSharp/blob/8f4def03b641b6753f18076aa5438f8eaaef2d30/src/TorchSharp/NN/Transformer.cs
- TorchSharp CUDA synchronize wrapper: https://github.com/dotnet/TorchSharp/blob/8f4def03b641b6753f18076aa5438f8eaaef2d30/src/TorchSharp/Torch.cs
- PyTorch SDPA semantics and dispatch caveats: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

Inspected source signature is not proof that the pinned NuGet dependency was compiled here. Restore/build and native parity remain Grok gates.
SDPA selects supported native kernels; no unconditional FlashAttention/memory-efficiency promise follows from calling this API in FP32.


## Audit5 semantics and verification scope

- PyTorch detach shares storage unless cloned: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html
- .NET TextReader chunked async API: https://learn.microsoft.com/en-us/dotnet/api/system.io.textreader.readasync
- TorchSharp dispose-scope source: https://github.com/dotnet/TorchSharp/blob/main/src/TorchSharp/DisposeScope.cs

The source rewrite uses existing pinned tensor APIs. No dependency version was upgraded in audit5. Actual pinned API compatibility
and autograd/dispose behavior are mandatory compilation/native tests for Grok; web documentation alone does not certify them.

## Audit6 additional implementation references

Microsoft .NET IncrementalHash class and AppendData/GetHashAndReset contracts:
https://learn.microsoft.com/en-us/dotnet/api/system.security.cryptography.incrementalhash?view=net-10.0
https://learn.microsoft.com/en-us/dotnet/api/system.security.cryptography.incrementalhash.appenddata?view=net-10.0
https://learn.microsoft.com/en-us/dotnet/api/system.security.cryptography.incrementalhash.gethashandreset?view=net-10.0
No dependency versions were upgraded. Source API compatibility and native behavior still require compilation/runtime gates.

## Audit7 inspected APIs

TorchSharp native SiLU wrapper (non-inplace path):
https://github.com/dotnet/TorchSharp/blob/8f4def03b641b6753f18076aa5438f8eaaef2d30/src/TorchSharp/NN/Activation/SiLU.cs
.NET FileStream and File.Move contracts:
https://learn.microsoft.com/en-us/dotnet/api/system.io.filestream?view=net-10.0
https://learn.microsoft.com/en-us/dotnet/api/system.io.file.move?view=net-10.0
Inspected through primary sources. Dependencies were not upgraded. No API-source inspection replaces the pinned NuGet build/runtime test.

## Audit9 primary API references, inspected2026-09-16

- https://learn.microsoft.com/en-us/dotnet/api/system.io.streamreader.-ctor?view=net-10.0
  BOM autodetection can replace the requested encoding; the new import reader handles only UTF8 BOM explicitly.
- https://learn.microsoft.com/en-us/dotnet/api/system.text.utf8encoding?view=net-10.0
  `UTF8Encoding(false,true)` enables malformed-byte rejection.
- https://learn.microsoft.com/en-us/dotnet/standard/serialization/system-text-json/how-to
  Stream serialization used by checkpoint JSON writing.

These documents establish API semantics, not a successful compilation or execution of the application.
