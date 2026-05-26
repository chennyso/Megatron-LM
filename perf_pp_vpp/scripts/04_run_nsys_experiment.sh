#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/03_run_megatron_experiment.sh" "$@" --profile-mode nsys
