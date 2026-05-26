#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: 11e_export_and_analyze_nsys.sh <remote-run-dir>" >&2
  exit 2
fi

RUN_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while IFS= read -r -d '' rep; do
  sqlite="${rep%.nsys-rep}.sqlite"
  json="${rep%.nsys-rep}.analysis.json"
  nsys export --type sqlite --force-overwrite=true --output "${sqlite}" "${rep}" >/dev/null
  python3 "${SCRIPT_DIR}/11d_analyze_nsys_sqlite.py" "${sqlite}" > "${json}"
  echo "${json}"
done < <(find "${RUN_DIR}" -type f -name '*.nsys-rep' -print0)
