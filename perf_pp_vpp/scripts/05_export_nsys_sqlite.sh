#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")"/../configs && pwd)/env.sh"

ROOT="${1:-${OUTPUT_ROOT}/runs}"
LOG="${OUTPUT_ROOT}/nsys_export_errors.log"
: > "${LOG}"

while IFS= read -r -d '' rep; do
  sqlite="${rep%.nsys-rep}.sqlite"
  if ! ${NSYS_BIN} export --type sqlite --force-overwrite=true --output "${sqlite}" "${rep}" >> "${LOG}" 2>&1; then
    echo "failed: ${rep}" >> "${LOG}"
  fi
done < <(find "${ROOT}" -type f -name '*.nsys-rep' -print0)
