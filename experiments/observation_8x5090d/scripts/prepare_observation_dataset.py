#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESS_SCRIPT = REPO_ROOT / "tools" / "preprocess_data.py"


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", required=True)
    parser.add_argument("--dataset-spec-id", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    matrix = load_matrix(Path(args.matrix_path))
    dataset_cfg = matrix["datasets"][args.dataset_spec_id]
    if dataset_cfg["kind"] != "real" or not dataset_cfg.get("hf_dataset"):
        raise SystemExit(f"Dataset spec {args.dataset_spec_id} is not an HF-backed real dataset.")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The `datasets` package is required to prepare observation datasets.") from exc

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_root / "raw.jsonl"
    output_prefix = output_root / output_root.name
    limit_docs = args.limit_docs or dataset_cfg.get("document_count", 50000)

    if raw_jsonl.exists() and not args.overwrite:
        raise SystemExit(f"{raw_jsonl} already exists. Pass --overwrite to rebuild.")

    ds = load_dataset(dataset_cfg["hf_dataset"], revision=dataset_cfg.get("revision"), split="train")
    selected = ds.select(range(min(limit_docs, len(ds))))
    with raw_jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")

    tokenizer_model = dataset_cfg.get("tokenizer_model") or next(
        model.get("tokenizer_model")
        for model in matrix["models"].values()
        if model.get("tokenizer_model")
    )

    cmd = [
        "python3",
        str(PREPROCESS_SCRIPT),
        "--input",
        str(raw_jsonl),
        "--output-prefix",
        str(output_prefix),
        "--dataset-impl",
        "mmap",
        "--tokenizer-type",
        dataset_cfg.get("tokenizer_type", "HuggingFaceTokenizer"),
        "--tokenizer-model",
        tokenizer_model,
        "--append-eod",
        "--workers",
        str(args.workers),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    manifest = {
        "dataset_spec_id": args.dataset_spec_id,
        "hf_dataset": dataset_cfg["hf_dataset"],
        "revision": dataset_cfg.get("revision"),
        "document_count": len(selected),
        "raw_jsonl": str(raw_jsonl),
        "output_prefix": str(output_prefix),
        "tokenizer_model": tokenizer_model,
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
