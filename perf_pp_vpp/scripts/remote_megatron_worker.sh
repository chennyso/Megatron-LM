#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <env-file> <command-file>"
  exit 2
fi

ENV_FILE="$1"
CMD_FILE="$2"
source "$ENV_FILE"
exec bash "$CMD_FILE"
