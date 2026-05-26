from __future__ import annotations

import csv
import json
import math
import os
import re
import shlex
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


PERF_SEED = int(os.environ.get("PERF_SEED", "1234"))
OVERLAP_MODES = (
    "baseline",
    "p2p_overlap",
    "dp_overlap",
    "all_overlap",
    "p2p_flush_overlap",
    "all_overlap_p2p_flush",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def perf_root() -> Path:
    return project_root() / "perf_pp_vpp"


def local_runs_root() -> Path:
    override = env_value("PERF_RUNS_ROOT_LOCAL", "").strip()
    if override:
        return Path(override)
    return perf_root() / "outputs" / "runs"


def remote_runs_root() -> Path:
    override = env_value("PERF_RUNS_ROOT_REMOTE", "").strip()
    if override:
        return Path(override)
    return Path(remote_root()) / "perf_pp_vpp" / "outputs" / "runs"


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        names = set()
        for row in rows:
            names.update(row.keys())
        fieldnames = list(sorted(names))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def sh(cmd: List[str], *, check: bool = True, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def run_text(cmd: List[str], *, check: bool = True, env: Optional[Dict[str, str]] = None, cwd: Optional[Path] = None) -> str:
    cp = sh(cmd, check=check, env=env, cwd=cwd)
    return (cp.stdout or "") + (cp.stderr or "")


def env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def remote_root() -> str:
    return env_value("REMOTE_PROJECT_ROOT", "/workspace/code/Megatron-LM")


def kubectl_cmd() -> List[str]:
    candidate = env_value("KUBECTL_BIN", "kubectl")
    tokens = shlex.split(candidate)
    if not tokens:
        return ["kubectl"]
    if tokens[0] == "kubectl" and Path("/usr/local/bin/kubectl").exists():
        return ["bash", "/usr/local/bin/kubectl", *tokens[1:]]
    path = Path(tokens[0])
    if path.exists():
        return ["bash", str(path), *tokens[1:]]
    return tokens


def kubectl_prefix(pod: str) -> List[str]:
    return kubectl_cmd() + ["exec", "-n", env_value("K8S_NAMESPACE", "default"), pod, "--", "bash", "-lc"]


def kubectl_cp_prefix() -> List[str]:
    return kubectl_cmd() + ["cp"]


def remote_cat_json(pod: str, path: str) -> Dict[str, Any]:
    text = run_text(kubectl_prefix(pod) + [f"test -f {path!r} && cat {path!r}"])
    return json.loads(text.strip())


def load_model_yaml(model_name: str) -> Dict[str, Any]:
    model_path = perf_root() / "configs" / "models" / f"{model_name}.yaml"
    if not model_path.exists():
        raise SystemExit(f"missing model yaml: {model_path}")
    return load_yaml(model_path)


def resolve_remote_model_config(model_name: str) -> Dict[str, Any]:
    model_yaml = load_model_yaml(model_name)
    env_var = model_yaml["source_env_var"]
    model_root = env_value(env_var, "")
    if not model_root:
        raise SystemExit(f"{env_var} is required for {model_name}")
    pod = env_value("NODE0_POD", "chenny-dist-0")
    config_path = f"{model_root.rstrip('/')}/config.json"
    text = run_text(kubectl_prefix(pod) + [f"test -f {config_path!r} && cat {config_path!r}"])
    config = json.loads(text.strip())
    return {
        "model_name": model_yaml["model_name"],
        "model_root": model_root,
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "vocab_size": int(config["vocab_size"]),
        "num_layers": int(config.get("num_hidden_layers", model_yaml["num_layers"])),
        "num_attention_heads": int(config.get("num_attention_heads", model_yaml["num_attention_heads"])),
        "num_query_groups": int(config.get("num_key_value_heads", model_yaml["num_query_groups"])),
        "seq_length_list": list(model_yaml["seq_length_list"]),
        "default_seq_length": int(model_yaml["default_seq_length"]),
        "max_position_embeddings": int(config.get("max_position_embeddings", model_yaml.get("max_position_embeddings", 32768))),
        "rotary_base": int(config.get("rope_theta", model_yaml.get("rotary_base", 1000000))),
    }


def world_size() -> int:
    return int(env_value("NNODES", "2")) * int(env_value("NPROC_PER_NODE", "8"))


def layers_per_virtual_stage(num_layers: int, pp: int, vpp: int) -> Optional[int]:
    denom = pp * vpp
    if denom <= 0 or num_layers % denom != 0:
        return None
    return num_layers // denom


def pipeline_layout_stage_count(layout: str) -> int:
    layout = layout.replace(",", "")
    patterns = [
        r'\(([^)]+)\)\*(\d+)',
        r'(.)\*(\d+)',
    ]
    for pattern in patterns:
        layout = re.sub(pattern, lambda x: x.group(1) * int(x.group(2)), layout)
    return len(layout.split('|'))


def derive_global_batch_size(micro_batch: int, dp: int) -> int:
    return micro_batch * dp * 16


def overlap_flags(mode: str) -> Dict[str, Any]:
    mode = mode or "baseline"
    if mode not in OVERLAP_MODES:
        raise SystemExit(f"unsupported overlap mode: {mode}")
    return {
        "baseline": {
            "overlap_p2p_comm": False,
            "overlap_grad_reduce": False,
            "overlap_param_gather": False,
        },
        "p2p_overlap": {
            "overlap_p2p_comm": True,
            "overlap_grad_reduce": False,
            "overlap_param_gather": False,
        },
        "dp_overlap": {
            "overlap_p2p_comm": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
        },
        "all_overlap": {
            "overlap_p2p_comm": True,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
        },
        "p2p_flush_overlap": {
            "overlap_p2p_comm": True,
            "overlap_grad_reduce": False,
            "overlap_param_gather": False,
        },
        "all_overlap_p2p_flush": {
            "overlap_p2p_comm": True,
            "overlap_grad_reduce": True,
            "overlap_param_gather": True,
        },
    }[mode]


def experiment_yaml(path: Path) -> Dict[str, Any]:
    payload = load_yaml(path)
    if "experiments" not in payload:
        raise SystemExit(f"experiment yaml missing experiments: {path}")
    return payload


def validate_experiment(base: Dict[str, Any], exp: Dict[str, Any], model: Dict[str, Any]) -> Tuple[bool, str, Optional[int]]:
    total = int(exp["pp"]) * int(exp["dp"]) * int(exp["tp"])
    if total != world_size():
        return False, f"world_size mismatch: pp*dp*tp={total} != {world_size()}", None
    vpp = max(int(exp.get("vpp", 1)), 1)
    num_virtual_stages = os.environ.get("PERF_NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK", "").strip()
    pipeline_layout = os.environ.get("PERF_PIPELINE_MODEL_PARALLEL_LAYOUT", "").strip()
    if pipeline_layout:
        num_stages = pipeline_layout_stage_count(pipeline_layout)
        if num_stages % int(exp["pp"]) != 0:
            return False, (
                f"pipeline layout stage count {num_stages} not divisible by pp={exp['pp']}"
            ), None
        layout_vpp = num_stages // int(exp["pp"])
        if layout_vpp != vpp:
            return False, (
                f"pipeline layout implies vpp={layout_vpp}, but experiment requests vpp={vpp}"
            ), None
        return True, "", None
    if num_virtual_stages:
        if int(num_virtual_stages) != vpp:
            return False, (
                f"PERF_NUM_VIRTUAL_STAGES_PER_PIPELINE_RANK={num_virtual_stages} "
                f"does not match experiment vpp={vpp}"
            ), None
        return True, "", None
    layers = layers_per_virtual_stage(int(model["num_layers"]), int(exp["pp"]), vpp)
    if layers is None:
        return False, f"num_layers={model['num_layers']} not divisible by pp*vpp={exp['pp']}*{vpp}", None
    return True, "", layers


def feasible_experiments(experiment_path: Path) -> Dict[str, Any]:
    payload = experiment_yaml(experiment_path)
    model = resolve_remote_model_config(payload["model"])
    feasible = []
    skipped = []
    for exp in payload["experiments"]:
        ok, reason, layers = validate_experiment(payload, exp, model)
        item = dict(exp)
        item["layers_per_virtual_stage"] = layers
        if ok:
            feasible.append(item)
        else:
            item["skip_reason"] = reason
            skipped.append(item)
    return {"base": payload, "model": model, "feasible": feasible, "skipped": skipped}


def resolved_experiment(experiment_path: Path, experiment_name: str, seq_len: int, repeat_id: int, overlap_mode: str) -> Dict[str, Any]:
    payload = feasible_experiments(experiment_path)
    matches = [x for x in payload["feasible"] if x["name"] == experiment_name]
    if not matches:
        skipped = {x["name"]: x["skip_reason"] for x in payload["skipped"]}
        if experiment_name in skipped:
            raise SystemExit(f"experiment skipped: {experiment_name}: {skipped[experiment_name]}")
        raise SystemExit(f"unknown experiment: {experiment_name}")
    exp = dict(matches[0])
    base = payload["base"]
    model = payload["model"]
    micro = int(env_value("PERF_MICRO_BATCH_SIZE", str(base["micro_batch_size"])))
    global_batch = int(
        env_value("PERF_GLOBAL_BATCH_SIZE", str(derive_global_batch_size(micro, int(exp["dp"]))))
    )
    overlap = overlap_flags(overlap_mode)
    run_name = env_value("PERF_RUN_NAME_OVERRIDE", exp["name"]).strip() or exp["name"]
    run_tag = env_value("PERF_RUN_TAG", "").strip()
    run_leaf = overlap_mode if not run_tag else f"{overlap_mode}__{run_tag}"
    run_dir = local_runs_root() / run_name / f"seq{seq_len}" / f"repeat{repeat_id}" / run_leaf
    remote_run_dir = remote_runs_root() / run_name / f"seq{seq_len}" / f"repeat{repeat_id}" / run_leaf
    return {
        "base": base,
        "model": model,
        "experiment": exp,
        "seq_len": int(seq_len),
        "repeat_id": int(repeat_id),
        "global_batch_size": global_batch,
        "micro_batch_size": micro,
        "overlap_mode": overlap_mode,
        "overlap": overlap,
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "remote_run_dir": str(remote_run_dir),
    }


def regex_float(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_step_records(lines: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in lines:
        if "iteration" not in line.lower():
            continue
        row: Dict[str, Any] = {"raw": line.rstrip("\n")}
        m = re.search(r"iteration\s+(\d+)", line, re.IGNORECASE)
        if m:
            row["iteration"] = int(m.group(1))
        row["elapsed_time_ms"] = regex_float(line, r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
        if row["elapsed_time_ms"] is None:
            row["elapsed_time_ms"] = regex_float(line, r"iteration-time[:= ]+([0-9.]+)")
        row["samples_per_second"] = regex_float(line, r"samples per second[:= ]+([0-9.]+)")
        row["tokens_per_second"] = regex_float(line, r"tokens per second[:= ]+([0-9.]+)")
        row["lm_loss"] = regex_float(line, r"lm loss[:= ]+([0-9.eE+-]+)")
        row["tflops"] = regex_float(line, r"TFLOP[s]?[:= ]+([0-9.]+)")
        if row["tflops"] is None:
            row["tflops"] = regex_float(line, r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9.]+)")
        row["mfu"] = regex_float(line, r"MFU[:= ]+([0-9.]+)")
        row["allocated_memory"] = regex_float(line, r"allocated memory[:= ]+([0-9.]+)")
        row["max_allocated_memory"] = regex_float(line, r"max allocated memory[:= ]+([0-9.]+)")
        row["reserved_memory"] = regex_float(line, r"reserved memory[:= ]+([0-9.]+)")
        if "iteration" in row:
            rows.append(row)
    return rows


def mean_std(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return None, None
    if len(valid) == 1:
        return valid[0], 0.0
    return statistics.mean(valid), statistics.stdev(valid)


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def sqlite_tables(path: Path) -> List[str]:
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("select name from sqlite_master where type='table'").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()
