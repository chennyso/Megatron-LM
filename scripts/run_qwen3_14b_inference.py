#!/usr/bin/env python3
"""
Utility launcher for running Qwen3-14B static text generation on 8x NVIDIA RTX 4090D GPUs.

This script inspects the HuggingFace/ModelScope config that lives inside
the checkpoint directory so that tensor/pipeline parallel arguments always
match the source model definition.  It then assembles the corresponding
`torchrun ... examples/inference/gpt/gpt_static_inference.py` command with
a set of defaults that keep memory usage below the 24 GB envelope while
enabling TE kernels, FlashAttention and tensor↔pipeline hybrid parallelism.

Example (single node):

    python scripts/run_qwen3_14b_inference.py \\
        --prompts "介绍一下Megatron-LM" "写一首七言绝句"
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


CHECKPOINT_ENV_DEFAULT = "/public/home/ssjxscy/.cache/modelscope/hub/models/Qwen/Qwen3-14B"
REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_ENTRYPOINT = REPO_ROOT / "examples" / "inference" / "gpt" / "gpt_static_inference.py"


def _load_qwen_config(model_dir: Path) -> Dict[str, Any]:
    """Load the HuggingFace/ModelScope config json from the supplied directory."""
    config_candidates = ("config.json", "model_config.json", "hf_config.json")
    for candidate in config_candidates:
        cfg_path = model_dir / candidate
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as reader:
                return json.load(reader)
    raise FileNotFoundError(
        f"Could not locate a config json in {model_dir}. "
        "Make sure the checkpoint directory is a HuggingFace/ModelScope export."
    )


def _extract_arch_fields(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map HF config keys to the Megatron CLI flags we will pass downstream."""

    def _find(keys: List[str], default: Optional[Any] = None):
        for key in keys:
            if key in cfg:
                return cfg[key]
        if default is None:
            raise KeyError(f"Missing required config entries {keys}")
        return default

    hidden_size = _find(["hidden_size", "n_embd"])
    num_heads = _find(["num_attention_heads", "num_heads", "n_head"])
    num_kv_heads = _find(["num_key_value_heads", "num_kv_heads", "n_kv_head"], default=num_heads)
    num_layers = _find(["num_hidden_layers", "n_layer", "num_layers"])
    intermediate_size = _find(["intermediate_size", "ffn_hidden_size"])
    max_seq_len = _find(["max_position_embeddings", "max_sequence_length"])
    vocab_size = _find(["vocab_size"])
    rope_theta = _find(["rope_theta", "rotary_base"], default=1000000)
    norm_eps = _find(["rms_norm_eps", "layer_norm_epsilon"], default=1e-6)
    rotary_pct = cfg.get("rope_scaling", {}).get("m", cfg.get("rotary_pct", 1.0))
    swiglu = cfg.get("use_swiglu", True)
    qk_layernorm = bool(cfg.get("use_qk_layernorm", True))
    kv_channels = hidden_size // num_heads

    return {
        "num_layers": int(num_layers),
        "hidden_size": int(hidden_size),
        "ffn_hidden_size": int(intermediate_size),
        "num_attention_heads": int(num_heads),
        "num_query_groups": int(num_kv_heads),
        "kv_channels": int(kv_channels),
        "max_position_embeddings": int(max_seq_len),
        "vocab_size": int(vocab_size),
        "rotary_base": float(rope_theta),
        "rotary_percent": float(rotary_pct),
        "norm_epsilon": float(norm_eps),
        "use_swiglu": bool(swiglu),
        "use_qk_layernorm": qk_layernorm,
    }


def _build_launcher_args(args, model_config: Dict[str, Any]) -> List[str]:
    """Compose the torchrun + Megatron command."""
    torchrun_cmd: List[str] = [
        "torchrun",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        str(INFERENCE_ENTRYPOINT),
    ]

    model_args = [
        "--load", args.model_path,
        "--ckpt-format", "torch_dist",
        "--use-flash-attn",
        "--transformer-impl", "transformer_engine",
        "--use-te",
        "--bf16",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--attention-softmax-in-fp32",
        "--no-masked-softmax-fusion",
        "--untie-embeddings-and-output-weights",
        "--disable-bias-linear",
        "--position-embedding-type", "rope",
        "--rotary-percent", str(model_config["rotary_percent"]),
        "--rotary-base", str(model_config["rotary_base"]),
        "--use-rotary-position-embeddings",
        "--swiglu",
        "--group-query-attention",
        "--tensor-model-parallel-size", str(args.tensor_model_parallel_size),
        "--pipeline-model-parallel-size", str(args.pipeline_model_parallel_size),
        "--context-parallel-size", str(args.context_parallel_size),
        "--micro-batch-size", str(args.micro_batch_size),
        "--inference-max-seq-length", str(args.inference_max_seq_length),
        "--inference-max-requests", str(args.inference_max_requests),
        "--seq-length", str(args.inference_max_seq_length),
        "--max-position-embeddings", str(model_config["max_position_embeddings"]),
        "--num-layers", str(model_config["num_layers"]),
        "--hidden-size", str(model_config["hidden_size"]),
        "--ffn-hidden-size", str(model_config["ffn_hidden_size"]),
        "--num-attention-heads", str(model_config["num_attention_heads"]),
        "--kv-channels", str(model_config["kv_channels"]),
        "--num-query-groups", str(model_config["num_query_groups"]),
        "--normalization", "RMSNorm",
        "--norm-epsilon", str(model_config["norm_epsilon"]),
        "--vocab-size", str(model_config["vocab_size"]),
        "--make-vocab-size-divisible-by", "128",
        "--tokenizer-type", "HuggingFaceTokenizer",
        "--tokenizer-model", args.tokenizer_path,
        "--temperature", str(args.temperature),
        "--top_k", str(args.top_k),
        "--top_p", str(args.top_p),
        "--num-tokens-to-generate", str(args.num_tokens_to_generate),
        "--distributed-timeout-minutes", str(args.distributed_timeout),
        "--exit-on-missing-checkpoint",
    ]

    if args.sequence_parallel:
        model_args.append("--sequence-parallel")
    if model_config["use_qk_layernorm"]:
        model_args.append("--qk-layernorm")
    if args.stream:
        model_args.append("--stream")
    if args.return_logprobs:
        model_args.append("--return-log-probs")
    if args.prompt_file is not None:
        model_args.extend(["--prompt-file", args.prompt_file])
        if args.prompt_file_num_truncate:
            model_args.extend(["--prompt-file-num-truncate", str(args.prompt_file_num_truncate)])
    elif args.prompts:
        model_args.append("--prompts")
        model_args.extend(args.prompts)
    else:
        raise ValueError("Either --prompts or --prompt-file must be supplied.")

    if args.output_path:
        model_args.extend(["--output-path", args.output_path])

    return torchrun_cmd + model_args


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid parallel Qwen3-14B inference launcher for Megatron-LM."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL_PATH", CHECKPOINT_ENV_DEFAULT),
        help="Directory that holds the sharded checkpoint (defaults to MODEL_PATH env variable).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=os.environ.get("TOKENIZER_PATH", CHECKPOINT_ENV_DEFAULT),
        help="Path passed to --tokenizer-model (defaults to MODEL_PATH).",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel degree. Recommended: 4 on 8x4090D.",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=2,
        help="Pipeline parallel degree. Recommended: 2 to fully utilize 8 GPUs.",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Context parallel degree (keep at 1 for single-node inference).",
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=1, help="Micro batch size per pipeline stage."
    )
    parser.add_argument(
        "--inference-max-seq-length",
        type=int,
        default=8192,
        help="Max total sequence length for static inference.",
    )
    parser.add_argument(
        "--inference-max-requests",
        type=int,
        default=4,
        help="Max number of concurrent requests for the static inference engine.",
    )
    parser.add_argument(
        "--sequence-parallel",
        dest="sequence_parallel",
        action="store_true",
        default=True,
        help="Enable sequence parallelism on tensor groups (recommended).",
    )
    parser.add_argument(
        "--no-sequence-parallel",
        dest="sequence_parallel",
        action="store_false",
        help="Disable sequence parallelism.",
    )
    parser.add_argument("--prompts", nargs="*", help="Inline prompts (mutually exclusive with --prompt-file).")
    parser.add_argument("--prompt-file", type=str, help="JSONL prompt file consumed by Megatron.")
    parser.add_argument(
        "--prompt-file-num-truncate",
        type=int,
        help="Truncate prompt file to this many samples before dispatch.",
    )
    parser.add_argument("--stream", action="store_true", help="Enable streaming generation.")
    parser.add_argument("--return-logprobs", action="store_true", help="Return log probabilities.")
    parser.add_argument("--output-path", type=str, help="Write generations to this file.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", dest="top_k", type=int, default=1)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.9)
    parser.add_argument("--num-tokens-to-generate", type=int, default=512)
    parser.add_argument("--distributed-timeout", type=int, default=60)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=6000)
    parser.add_argument("--dry-run", action="store_true", help="Only print the composed command.")
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    args.model_path = str(model_path)
    tokenizer_path = Path(args.tokenizer_path).expanduser().resolve()
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path {tokenizer_path} does not exist.")
    args.tokenizer_path = str(tokenizer_path)

    world_size = (
        args.tensor_model_parallel_size
        * args.pipeline_model_parallel_size
        * args.context_parallel_size
    )
    if args.nproc_per_node != world_size:
        raise ValueError(
            f"--nproc-per-node ({args.nproc_per_node}) must equal TP x PP x CP ({world_size}) "
            "when running single-node inference."
        )

    hf_cfg = _load_qwen_config(model_path)
    model_fields = _extract_arch_fields(hf_cfg)

    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NVTE_APPLY_QK_LAYER_SCALING", "0")

    launcher_cmd = _build_launcher_args(args, model_fields)
    printable_cmd = " ".join(shlex.quote(arg) for arg in launcher_cmd)
    print(f"[Megatron Launcher] {printable_cmd}")

    if args.dry_run:
        return

    subprocess.run(launcher_cmd, check=True)


if __name__ == "__main__":
    main()
