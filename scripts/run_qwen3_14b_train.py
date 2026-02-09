#!/usr/bin/env python3
# Copyright (c) 2025
"""
单机 8×RTX 4090D 混合并行微调 / 继续预训练 Qwen3-14B 的启动脚本。

脚本会自动读取 HuggingFace/ModelScope checkpoint 里的 config.json，
生成符合 Megatron-LM `pretrain_gpt.py` 所需的张量/流水/序列并行参数，
并添加一套对 24GB 显存友好的训练超参（recompute + sequence parallel 等）。

示例：
    python scripts/run_qwen3_14b_train.py \\
        --data-path /path/to/my_gpt_text_document \\
        --output-dir /workspace/ckpts/qwen3_14b_sft \\
        --train-iters 2000 --lr 2e-5 --lr-warmup-iters 200
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_PATH = "/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen3-14B"
REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ENTRYPOINT = REPO_ROOT / "pretrain_gpt.py"


def _load_model_config(model_dir: Path) -> Dict[str, Any]:
    """Locate并读取 config.json（或兼容命名）。"""
    for candidate in ("config.json", "model_config.json", "hf_config.json"):
        cfg = model_dir / candidate
        if cfg.exists():
            with cfg.open("r", encoding="utf-8") as reader:
                return json.load(reader)
    raise FileNotFoundError(f"在 {model_dir} 下没有找到 config.json / model_config.json。")


def _extract_arch(config: Dict[str, Any]) -> Dict[str, Any]:
    """从 HF 配置映射出 Megatron 关键结构参数。"""

    def _get(keys, default: Optional[Any] = None):
        for key in keys:
            if key in config:
                return config[key]
        if default is None:
            raise KeyError(f"缺少必须的配置字段：{keys}")
        return default

    hidden_size = _get(["hidden_size", "n_embd"])
    num_layers = _get(["num_hidden_layers", "num_layers", "n_layer"])
    num_heads = _get(["num_attention_heads", "n_head"])
    num_kv_heads = _get(["num_key_value_heads", "num_kv_heads"], default=num_heads)
    intermediate = _get(["intermediate_size", "ffn_hidden_size"])
    max_seq = _get(["max_position_embeddings", "max_sequence_length"])
    vocab_size = _get(["vocab_size"])
    rope_theta = config.get("rope_theta", config.get("rotary_base", 1000000))
    rotary_pct = config.get("rotary_pct", 1.0)
    norm_eps = config.get("rms_norm_eps", config.get("layer_norm_epsilon", 1e-6))
    kv_channels = hidden_size // num_heads
    swiglu = config.get("use_swiglu", True)
    qk_layernorm = config.get("use_qk_layernorm", True)

    return {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "num_query_groups": int(num_kv_heads),
        "kv_channels": int(kv_channels),
        "ffn_hidden_size": int(intermediate),
        "max_position_embeddings": int(max_seq),
        "vocab_size": int(vocab_size),
        "rotary_base": float(rope_theta),
        "rotary_percent": float(rotary_pct),
        "norm_epsilon": float(norm_eps),
        "use_swiglu": bool(swiglu),
        "use_qk_layernorm": bool(qk_layernorm),
    }


def _build_training_cmd(args, arch: Dict[str, Any]) -> List[str]:
    """组装 torchrun + pretrain_gpt.py 的完整命令。"""
    torchrun = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        str(PRETRAIN_ENTRYPOINT),
    ]

    model_flags = [
        "--tensor-model-parallel-size", str(args.tensor_model_parallel_size),
        "--pipeline-model-parallel-size", str(args.pipeline_model_parallel_size),
        "--context-parallel-size", str(args.context_parallel_size),
        "--num-layers", str(arch["num_layers"]),
        "--hidden-size", str(arch["hidden_size"]),
        "--ffn-hidden-size", str(arch["ffn_hidden_size"]),
        "--num-attention-heads", str(arch["num_heads"]),
        "--kv-channels", str(arch["kv_channels"]),
        "--group-query-attention",
        "--num-query-groups", str(arch["num_query_groups"]),
        "--normalization", "RMSNorm",
        "--norm-epsilon", str(arch["norm_epsilon"]),
        "--position-embedding-type", "rope",
        "--rotary-base", str(int(arch["rotary_base"])),
        "--rotary-percent", str(arch["rotary_percent"]),
        "--use-rotary-position-embeddings",
        "--max-position-embeddings", str(max(args.seq_length, arch["max_position_embeddings"])),
    ]
    if arch["use_swiglu"]:
        model_flags.append("--swiglu")
    if arch["use_qk_layernorm"]:
        model_flags.append("--qk-layernorm")

    training_flags = [
        "--micro-batch-size", str(args.micro_batch_size),
        "--global-batch-size", str(args.global_batch_size),
        "--seq-length", str(args.seq_length),
        "--max-position-embeddings", str(max(args.seq_length, arch["max_position_embeddings"])),
        "--train-iters", str(args.train_iters),
        "--lr", str(args.lr),
        "--min-lr", str(args.min_lr),
        "--lr-decay-style", args.lr_decay_style,
        "--lr-decay-iters", str(args.lr_decay_iters),
        "--lr-warmup-iters", str(args.lr_warmup_iters),
        "--weight-decay", str(args.weight_decay),
        "--clip-grad", str(args.clip_grad),
        "--adam-beta1", str(args.adam_beta1),
        "--adam-beta2", str(args.adam_beta2),
        "--adam-eps", str(args.adam_eps),
        "--data-path", args.data_path,
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", args.tokenizer_path,
        "--reset-position-ids",
        "--reset-attention-mask",
        "--bf16",
        "--transformer-impl", "transformer_engine",
        "--use-flash-attn",
        "--attention-softmax-in-fp32",
        "--no-masked-softmax-fusion",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--recompute-activations",
        "--recompute-granularity", "selective",
        "--recompute-num-layers", str(arch["num_layers"]),
        "--save", args.output_dir,
        "--save-interval", str(args.save_interval),
        "--eval-iters", str(args.eval_iters),
        "--eval-interval", str(args.eval_interval),
        "--log-throughput",
        "--load", args.model_path,
        "--no-load-optim",
        "--no-load-rng",
    ]
    if args.pack_data:
        training_flags.append("--pack-dataset")
    if args.sequence_parallel:
        training_flags.append("--sequence-parallel")
    if args.context_parallel_size > 1:
        training_flags.extend(["--context-parallel-algo", "ring_attention"])

    return torchrun + model_flags + training_flags


def main():
    parser = argparse.ArgumentParser(description="Qwen3-14B 混合并行训练启动脚本")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
                        help="Megatron/ModelScope checkpoint 根目录（默认读取 MODEL_PATH）")
    parser.add_argument("--tokenizer-path", default=os.environ.get("TOKENIZER_PATH", DEFAULT_MODEL_PATH),
                        help="Tokenizer 路径，默认为 MODEL_PATH")
    parser.add_argument("--data-path", default=os.environ.get("DATA_PATH", ""),
                        help="Megatron 预处理数据 .bin/.idx 前缀（例如 corpora/my_text_document），必填")
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./checkpoints/qwen3_14b"),
                        help="训练 checkpoint/日志输出目录")

    parser.add_argument("--tensor-model-parallel-size", type=int, default=4)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=2)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true", default=True)
    parser.add_argument("--no-sequence-parallel", dest="sequence_parallel", action="store_false")

    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--pack-data", action="store_true")

    parser.add_argument("--train-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--lr-decay-style", default="cosine")
    parser.add_argument("--lr-decay-iters", type=int, default=1000)
    parser.add_argument("--lr-warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--adam-eps", type=float, default=1e-8)

    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)

    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不真正执行")

    args = parser.parse_args()

    if not args.data_path:
        raise ValueError("必须通过 --data-path 或 DATA_PATH 环境变量提供 Megatron 预处理数据前缀。")

    model_path = Path(args.model_path).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"模型目录 {model_path} 不存在。")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer 路径 {tokenizer_path} 不存在。")

    args.model_path = str(model_path)
    args.tokenizer_path = str(tokenizer_path)
    args.output_dir = str(output_dir)

    world_size = (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.context_parallel_size
    )
    if args.nproc_per_node != world_size:
        raise ValueError(
            f"--nproc-per-node ({args.nproc_per_node}) 必须等于 TP×PP×CP ({world_size}) 才能单机运行。"
        )

    hf_config = _load_model_config(model_path)
    arch = _extract_arch(hf_config)

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("NVTE_APPLY_QK_LAYER_SCALING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cmd = _build_training_cmd(args, arch)
    printable_cmd = " ".join(shlex.quote(part) for part in cmd)
    print(f"[Qwen3-14B Train Launcher]\n{printable_cmd}")
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
