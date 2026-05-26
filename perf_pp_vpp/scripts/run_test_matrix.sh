#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/../configs/experiments/phase2_qwen32b_tp2_test_matrix.yaml"

echo "=========================================="
echo "Test Matrix: q32 TP2 PP/VPP Comparison"
echo "=========================================="

# Define test configurations
declare -A TESTS=(
  ["1_vpp1"]="q32_pp4_dp2_tp2_vpp1"
  ["2_vpp2"]="q32_pp4_dp2_tp2_vpp2"
  ["3_vpp4_baseline"]="q32_pp4_dp2_tp2_vpp4"
  ["4_vpp2_mbs2"]="q32_pp4_dp2_tp2_vpp2_mbs2"
  ["5_tp1_pp8_vpp1"]="q32_pp8_dp2_tp1_vpp1"
  ["6_tp1_pp8_vpp2"]="q32_pp8_dp2_tp1_vpp2"
  ["7_tp4_pp2_vpp1"]="q32_pp2_dp2_tp4_vpp1"
  ["8_tp4_pp2_vpp2"]="q32_pp2_dp2_tp4_vpp2"
)

# Run each test
for test_id in $(echo "${!TESTS[@]}" | tr ' ' '\n' | sort); do
  test_name="${TESTS[$test_id]}"
  echo ""
  echo "=========================================="
  echo "Running: ${test_id} (${test_name})"
  echo "=========================================="
  
  bash "${SCRIPT_DIR}/03_run_megatron_experiment.sh" \
    --experiment-yaml "${CONFIG}" \
    --experiment-name "${test_name}" \
    --seq-len 4096 \
    --repeat-id 1 \
    --profile-mode none \
    --profile-ranks none \
    --overlap-mode baseline \
    2>&1 | tee "${test_name}_output.log"
  
  echo "Completed: ${test_name}"
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
