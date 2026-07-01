#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path


def run_capture(cmd: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=False, stdout=handle, stderr=subprocess.STDOUT, text=True)


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_gpu_count() -> int:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return len([line for line in result.stdout.splitlines() if line.strip()])


def benchmark_pair(src: int, dst: int, num_bytes: int, iterations: int) -> tuple[float, float, int]:
    import torch

    can_access = 0
    if src != dst and hasattr(torch.cuda, "can_device_access_peer"):
        can_access = int(bool(torch.cuda.can_device_access_peer(src, dst)))

    if src == dst:
        return math.nan, math.nan, can_access

    src_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=f"cuda:{src}")
    dst_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=f"cuda:{dst}")

    for _ in range(10):
        dst_tensor.copy_(src_tensor, non_blocking=True)
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        dst_tensor.copy_(src_tensor, non_blocking=True)
    end.record()
    torch.cuda.synchronize(dst)

    ms_per_iter = start.elapsed_time(end) / iterations
    gbps = (num_bytes / (1024 ** 3)) / (ms_per_iter / 1000.0)
    return gbps, ms_per_iter * 1000.0, can_access


def run_pairwise_p2p(output_dir: Path, message_sizes: list[int], gpu_count: int) -> Path:
    output_path = output_dir / "pairwise_p2p.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["src_gpu", "dst_gpu", "num_bytes", "bandwidth_gbps", "latency_us", "peer_access"],
        )
        writer.writeheader()
        for num_bytes in message_sizes:
            iterations = 50 if num_bytes <= 64 * 1024 * 1024 else 10
            for src in range(gpu_count):
                for dst in range(gpu_count):
                    gbps, latency_us, can_access = benchmark_pair(src, dst, num_bytes, iterations)
                    writer.writerow(
                        {
                            "src_gpu": src,
                            "dst_gpu": dst,
                            "num_bytes": num_bytes,
                            "bandwidth_gbps": f"{gbps:.6f}" if not math.isnan(gbps) else "",
                            "latency_us": f"{latency_us:.3f}" if not math.isnan(latency_us) else "",
                            "peer_access": can_access,
                        }
                    )
    return output_path


def ensure_nccl_tests(workdir: Path) -> Path:
    repo_dir = workdir / "nccl-tests"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "https://github.com/NVIDIA/nccl-tests.git", str(repo_dir)],
            check=True,
        )
    binary = repo_dir / "build" / "all_reduce_perf"
    if binary.exists():
        return repo_dir
    make_env = os.environ.copy()
    subprocess.run(
        ["make", "-j", "MPI=0"],
        cwd=repo_dir,
        env=make_env,
        check=True,
    )
    return repo_dir


def parse_nccl_stdout(raw_path: Path, collective: str, gpu_count: int, num_bytes: int) -> list[dict]:
    rows: list[dict] = []
    for line in raw_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        tokens = stripped.split()
        if not tokens[0].isdigit():
            continue
        numeric_tokens = []
        for token in tokens:
            try:
                numeric_tokens.append(float(token))
            except ValueError:
                continue
        if len(numeric_tokens) < 4:
            continue
        rows.append(
            {
                "collective": collective,
                "gpu_count": gpu_count,
                "num_bytes": num_bytes,
                "reported_size": int(numeric_tokens[0]),
                "time_value": numeric_tokens[-4],
                "algbw_gbps": numeric_tokens[-3],
                "busbw_gbps": numeric_tokens[-2],
            }
        )
    return rows


def run_nccl_tests(output_dir: Path, matrix: dict, gpu_count: int) -> tuple[Path, list[Path]]:
    tools_dir = output_dir / "_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = ensure_nccl_tests(tools_dir)
    csv_path = output_dir / "nccl_collectives.csv"
    raw_paths: list[Path] = []
    binaries = {
        "all_reduce": "all_reduce_perf",
        "all_gather": "all_gather_perf",
        "reduce_scatter": "reduce_scatter_perf",
        "all_to_all": "alltoall_perf",
    }
    rows: list[dict] = []
    for collective in matrix["hardware"]["collectives"]:
        binary = repo_dir / "build" / binaries[collective]
        for n_gpus in matrix["hardware"]["gpu_counts"]:
            if n_gpus > gpu_count:
                continue
            for num_bytes in matrix["hardware"]["message_sizes_bytes"]:
                raw_path = output_dir / "raw" / f"{collective}_g{n_gpus}_{num_bytes}.log"
                raw_path.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(binary),
                    "-b",
                    str(num_bytes),
                    "-e",
                    str(num_bytes),
                    "-f",
                    "2",
                    "-g",
                    str(n_gpus),
                    "-n",
                    "20",
                    "-w",
                    "5",
                ]
                run_capture(cmd, raw_path)
                raw_paths.append(raw_path)
                rows.extend(parse_nccl_stdout(raw_path, collective, n_gpus, num_bytes))
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["collective", "gpu_count", "num_bytes", "reported_size", "time_value", "algbw_gbps", "busbw_gbps"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, raw_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    matrix = load_matrix(Path(args.matrix_path))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_capture(["nvidia-smi"], output_dir / "nvidia-smi.txt")
    run_capture(["nvidia-smi", "topo", "-m"], output_dir / "nvidia-smi-topo.txt")
    run_capture(["nvidia-smi", "nvlink", "-s"], output_dir / "nvidia-smi-nvlink.txt")
    run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv",
        ],
        output_dir / "nvidia-smi-query.csv",
    )
    run_capture(["lscpu"], output_dir / "lscpu.txt")
    run_capture(["numactl", "--hardware"], output_dir / "numactl-hardware.txt")

    gpu_count = detect_gpu_count()
    p2p_csv = run_pairwise_p2p(output_dir, matrix["hardware"]["message_sizes_bytes"], gpu_count)
    nccl_csv, raw_nccl_logs = run_nccl_tests(output_dir, matrix, gpu_count)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu_count": gpu_count,
        "files": {
            "nvidia_smi": "nvidia-smi.txt",
            "topology": "nvidia-smi-topo.txt",
            "nvlink": "nvidia-smi-nvlink.txt",
            "gpu_query": "nvidia-smi-query.csv",
            "lscpu": "lscpu.txt",
            "numactl": "numactl-hardware.txt",
            "pairwise_p2p": str(p2p_csv.relative_to(output_dir)),
            "nccl_summary": str(nccl_csv.relative_to(output_dir)),
            "nccl_raw_count": len(raw_nccl_logs)
        }
    }
    (output_dir / "hardware_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
