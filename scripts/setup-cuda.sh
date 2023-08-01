#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

if command -v nvidia-smi > /dev/null; then
  conda install --channel="nvidia" cuda
fi
