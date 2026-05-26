#!/usr/bin/env bash
set -euo pipefail

# Megatron Test Matrix Runner using Volcano Jobs
# Usage: bash run_test_matrix_volcano.sh [test_id|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/../k8s/megatron-test-matrix-template.yaml"
OUTPUT_DIR="/tmp/megatron-test-jobs"

mkdir -p "${OUTPUT_DIR}"

# Test configurations: name tp pp vpp mbs seq
declare -a TESTS=(
    "tp2_pp4_vpp1_mbs1:2:4:1:1:4096"
    "tp2_pp4_vpp2_mbs1:2:4:2:1:4096"
    "tp2_pp4_vpp4_mbs1:2:4:4:1:4096"
    "tp2_pp4_vpp2_mbs2:2:4:2:2:4096"
    "tp1_pp8_vpp1:1:8:1:1:4096"
    "tp1_pp8_vpp2:1:8:2:1:4096"
    "tp4_pp2_vpp1:4:2:1:1:4096"
    "tp4_pp2_vpp2:4:2:2:1:4096"
)

generate_job() {
    local test_name="$1" tp="$2" pp="$3" vpp="$4" mbs="$5" seq="$6"
    local output_file="${OUTPUT_DIR}/${test_name}.yaml"
    
    sed -e "s/\${TEST_NAME}/${test_name}/g" \
        -e "s/\${TP}/${tp}/g" \
        -e "s/\${PP}/${pp}/g" \
        -e "s/\${VPP}/${vpp}/g" \
        -e "s/\${MBS}/${mbs}/g" \
        -e "s/\${SEQ}/${seq}/g" \
        "${TEMPLATE}" > "${output_file}"
    
    echo "Generated: ${output_file}"
}

run_test() {
    local test_name="$1"
    local job_file="${OUTPUT_DIR}/${test_name}.yaml"
    
    if [[ ! -f "${job_file}" ]]; then
        echo "ERROR: Job file not found: ${job_file}" >&2
        return 1
    fi
    
    echo "=========================================="
    echo "Running: ${test_name}"
    echo "=========================================="
    
    # Delete existing job if any
    kubectl delete job "megatron-test-${test_name}" -n default 2>/dev/null || true
    
    # Apply job
    kubectl apply -f "${job_file}"
    
    # Wait for job to complete
    echo "Waiting for job to complete..."
    kubectl wait --for=condition=complete --timeout=600s job/megatron-test-${test_name} -n default
    
    # Get logs
    echo "Collecting logs..."
    kubectl logs -l app=megatron-test -n default --tail=100 > "${OUTPUT_DIR}/${test_name}_logs.txt" 2>&1
    
    echo "Completed: ${test_name}"
    echo "Logs: ${OUTPUT_DIR}/${test_name}_logs.txt"
}

# Main
TEST_ID="${1:-all}"

if [[ "${TEST_ID}" == "all" ]]; then
    # Generate all jobs
    for test in "${TESTS[@]}"; do
        IFS=':' read -r name tp pp vpp mbs seq <<< "${test}"
        generate_job "${name}" "${tp}" "${pp}" "${vpp}" "${mbs}" "${seq}"
    done
    
    echo ""
    echo "Generated all jobs in ${OUTPUT_DIR}"
    echo "To run a specific test: kubectl apply -f ${OUTPUT_DIR}/<test_name>.yaml"
    echo "To run all tests: bash $0 run_all"
    
elif [[ "${TEST_ID}" == "run_all" ]]; then
    # Run all tests sequentially
    for test in "${TESTS[@]}"; do
        IFS=':' read -r name tp pp vpp mbs seq <<< "${test}"
        run_test "${name}"
    done
    
else
    # Generate and run specific test
    idx=$((TEST_ID - 1))
    if [[ ${idx} -ge 0 && ${idx} -lt ${#TESTS[@]} ]]; then
        IFS=':' read -r name tp pp vpp mbs seq <<< "${TESTS[$idx]}"
        generate_job "${name}" "${tp}" "${pp}" "${vpp}" "${mbs}" "${seq}"
        run_test "${name}"
    else
        echo "Invalid test ID: ${TEST_ID}" >&2
        echo "Valid range: 1-${#TESTS[@]}" >&2
        exit 1
    fi
fi
