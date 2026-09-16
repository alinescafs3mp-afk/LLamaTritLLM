#!/usr/bin/env bash
set -euo pipefail
rid="${1:-win-x64}"
backend="${2:-full}"
args=(--target "$rid")
case "$backend" in cpu) args+=(--cpu-only);; full|cuda-windows|cuda-linux) args+=(--with-cuda);; *) echo "Unknown backend: $backend" >&2; exit 2;; esac
if [[ "${3:-}" == "--full" ]]; then args+=(--full); fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/deliver.sh" "${args[@]}"
