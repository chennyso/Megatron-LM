#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from stats_utils import summarize_numeric


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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(
    rows: list[dict],
    group_keys: list[str],
    metric_names: list[str],
    extra_keys: list[str] | None = None,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)

    summaries: list[dict] = []
    for key, bucket in sorted(grouped.items()):
        summary = {name: value for name, value in zip(group_keys, key)}
        for extra_key in extra_keys or []:
            summary[extra_key] = bucket[0][extra_key]
        for metric_name in metric_names:
            metric_summary = summarize_numeric(row.get(metric_name) for row in bucket)
            for stat_name, stat_value in metric_summary.items():
                summary[f"{metric_name}_{stat_name}"] = stat_value
        summaries.append(summary)
    return summaries


def benchmark_pair(
    src: int,
    dst: int,
    num_bytes: int,
    warmup_iterations: int,
    measured_iterations: int,
    repetitions: int,
) -> list[dict]:
    import torch

    peer_access = 0
    if src != dst and hasattr(torch.cuda, "can_device_access_peer"):
        peer_access = int(bool(torch.cuda.can_device_access_peer(src, dst)))

    if src == dst:
        return []

    with torch.cuda.device(src):
        src_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=f"cuda:{src}")
    with torch.cuda.device(dst):
        dst_tensor = torch.empty(num_bytes, dtype=torch.uint8, device=f"cuda:{dst}")
        stream = torch.cuda.Stream(device=dst)

    def issue_copy(copy_iterations: int) -> None:
        with torch.cuda.device(dst), torch.cuda.stream(stream):
            for _ in range(copy_iterations):
                dst_tensor.copy_(src_tensor, non_blocking=True)

    issue_copy(warmup_iterations)
    stream.synchronize()
    torch.cuda.synchronize(src)
    torch.cuda.synchronize(dst)

    rows: list[dict] = []
    for repeat_index in range(1, repetitions + 1):
        with torch.cuda.device(dst):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(measured_iterations):
                    dst_tensor.copy_(src_tensor, non_blocking=True)
                end.record(stream)
            end.synchronize()
            ms_per_iter = start.elapsed_time(end) / measured_iterations
        bandwidth_gbps = (num_bytes / (1024 ** 3)) / (ms_per_iter / 1000.0)
        rows.append(
            {
                "src_gpu": src,
                "dst_gpu": dst,
                "num_bytes": num_bytes,
                "repeat_index": repeat_index,
                "bandwidth_gbps": bandwidth_gbps,
                "latency_us": ms_per_iter * 1000.0,
                "peer_access": peer_access,
                "warmup_iterations": warmup_iterations,
                "measured_iterations": measured_iterations,
            }
        )
    return rows


def run_pairwise_p2p(output_dir: Path, matrix: dict, gpu_count: int) -> tuple[Path, Path]:
    cfg = matrix["hardware"]
    raw_rows: list[dict] = []
    for num_bytes in cfg["message_sizes_bytes"]:
        measured_iterations = 50 if num_bytes <= 64 * 1024 * 1024 else 10
        for src in range(gpu_count):
            for dst in range(gpu_count):
                raw_rows.extend(
                    benchmark_pair(
                        src=src,
                        dst=dst,
                        num_bytes=num_bytes,
                        warmup_iterations=cfg["warmup_iterations"],
                        measured_iterations=measured_iterations,
                        repetitions=cfg["repetitions"],
                    )
                )

    raw_path = output_dir / "pairwise_p2p_raw.csv"
    summary_path = output_dir / "pairwise_p2p_summary.csv"
    write_csv(
        raw_path,
        raw_rows,
        [
            "src_gpu",
            "dst_gpu",
            "num_bytes",
            "repeat_index",
            "bandwidth_gbps",
            "latency_us",
            "peer_access",
            "warmup_iterations",
            "measured_iterations",
        ],
    )
    summary_rows = summarize_rows(
        raw_rows,
        group_keys=["src_gpu", "dst_gpu", "num_bytes"],
        metric_names=["bandwidth_gbps", "latency_us"],
        extra_keys=["peer_access", "warmup_iterations", "measured_iterations"],
    )
    if summary_rows:
        write_csv(summary_path, summary_rows, list(summary_rows[0].keys()))
    return raw_path, summary_path


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
    subprocess.run(
        ["make", "-j", "MPI=0"],
        cwd=repo_dir,
        env=os.environ.copy(),
        check=True,
    )
    return repo_dir


def parse_nccl_stdout(raw_path: Path, collective: str, gpu_count: int, num_bytes: int, repeat_index: int) -> list[dict]:
    rows: list[dict] = []
    for line in raw_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or not stripped[0].isdigit():
            continue
        tokens = stripped.split()
        numeric_tokens: list[float] = []
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
                "repeat_index": repeat_index,
                "reported_size": int(numeric_tokens[0]),
                "time_value": numeric_tokens[-4],
                "algbw_gbps": numeric_tokens[-3],
                "busbw_gbps": numeric_tokens[-2],
            }
        )
    return rows


def run_nccl_tests(output_dir: Path, matrix: dict, gpu_count: int) -> tuple[Path, Path, list[Path]]:
    cfg = matrix["hardware"]
    tools_dir = output_dir / "_tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = ensure_nccl_tests(tools_dir)
    raw_csv_path = output_dir / "nccl_collectives_raw.csv"
    summary_csv_path = output_dir / "nccl_collectives_summary.csv"
    raw_paths: list[Path] = []
    rows: list[dict] = []
    binaries = {
        "all_reduce": "all_reduce_perf",
        "all_gather": "all_gather_perf",
        "reduce_scatter": "reduce_scatter_perf",
        "all_to_all": "alltoall_perf",
    }

    for collective in cfg["collectives"]:
        binary = repo_dir / "build" / binaries[collective]
        for n_gpus in cfg["gpu_counts"]:
            if n_gpus > gpu_count:
                continue
            for num_bytes in cfg["message_sizes_bytes"]:
                for repeat_index in range(1, cfg["repetitions"] + 1):
                    raw_path = output_dir / "raw" / f"{collective}_g{n_gpus}_{num_bytes}_r{repeat_index:02d}.log"
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
                        str(cfg["measure_iterations"]),
                        "-w",
                        str(cfg["warmup_iterations"]),
                    ]
                    run_capture(cmd, raw_path)
                    raw_paths.append(raw_path)
                    rows.extend(parse_nccl_stdout(raw_path, collective, n_gpus, num_bytes, repeat_index))

    write_csv(
        raw_csv_path,
        rows,
        [
            "collective",
            "gpu_count",
            "num_bytes",
            "repeat_index",
            "reported_size",
            "time_value",
            "algbw_gbps",
            "busbw_gbps",
        ],
    )
    summary_rows = summarize_rows(
        rows,
        group_keys=["collective", "gpu_count", "num_bytes"],
        metric_names=["time_value", "algbw_gbps", "busbw_gbps"],
        extra_keys=["reported_size"],
    )
    if summary_rows:
        write_csv(summary_csv_path, summary_rows, list(summary_rows[0].keys()))
    return raw_csv_path, summary_csv_path, raw_paths


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
    p2p_raw_csv, p2p_summary_csv = run_pairwise_p2p(output_dir, matrix, gpu_count)
    nccl_raw_csv, nccl_summary_csv, raw_nccl_logs = run_nccl_tests(output_dir, matrix, gpu_count)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu_count": gpu_count,
        "statistics_policy": {
            "repetitions": matrix["hardware"]["repetitions"],
            "warmup_iterations": matrix["hardware"]["warmup_iterations"],
            "measure_iterations": matrix["hardware"]["measure_iterations"],
            "reported_statistics": ["mean", "std", "95% CI"],
        },
        "files": {
            "nvidia_smi": "nvidia-smi.txt",
            "topology": "nvidia-smi-topo.txt",
            "nvlink": "nvidia-smi-nvlink.txt",
            "gpu_query": "nvidia-smi-query.csv",
            "lscpu": "lscpu.txt",
            "numactl": "numactl-hardware.txt",
            "pairwise_p2p_raw": str(p2p_raw_csv.relative_to(output_dir)),
            "pairwise_p2p_summary": str(p2p_summary_csv.relative_to(output_dir)),
            "nccl_raw": str(nccl_raw_csv.relative_to(output_dir)),
            "nccl_summary": str(nccl_summary_csv.relative_to(output_dir)),
            "nccl_raw_count": len(raw_nccl_logs),
        },
    }
    (output_dir / "hardware_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
