#!/usr/bin/env bash
set -euo pipefail
. /etc/os-release
if [[ "${ID:-}" != ubuntu ]]; then
  echo "This script targets Ubuntu. Install .NET 10 SDK and Avalonia desktop dependencies using your distribution's documentation." >&2
  exit 2
fi
if [[ "${VERSION_ID:-}" == "22.04" ]]; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:dotnet/backports
fi
sudo apt-get update
sudo apt-get install -y dotnet-sdk-10.0 git curl unzip \
  libx11-6 libice6 libsm6 libfontconfig1 libxext6 libxrender1 \
  libxrandr2 libxi6 libxcursor1 libgl1 libgomp1
# The SDK pulls in its runtime/dependency packages. No Python or CUDA compiler is needed for the app.
dotnet --info
printf '\nReady for CPU build. NVIDIA drivers are deliberately not installed or modified.\n'
