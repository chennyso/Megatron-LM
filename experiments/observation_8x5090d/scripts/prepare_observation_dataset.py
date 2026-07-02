#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
PREPROCESS_SCRIPT = REPO_ROOT / "tools" / "preprocess_data.py"


def load_matrix(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def preprocess_output_prefix(data_path: Path) -> Path:
    suffix = "_text_document"
    data_path_text = str(data_path)
    if data_path_text.endswith(suffix):
        return Path(data_path_text[: -len(suffix)])
    return data_path


def load_fineweb_docs_via_mirror(dataset_cfg: dict, limit_docs: int, output_root: Path) -> tuple[int, dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("The `pyarrow` package is required for mirror-backed parquet dataset preparation.") from exc

    endpoint = dataset_cfg.get("hf_endpoint", "https://hf-mirror.com").rstrip("/")
    dataset_id = dataset_cfg["hf_dataset"]
    revision = dataset_cfg.get("revision", "main")
    path_prefix = dataset_cfg.get("hf_path_prefix", "data/")
    metadata_url = f"{endpoint}/api/datasets/{dataset_id}"
    metadata = requests.get(metadata_url, timeout=60).json()
    resolved_revision = metadata.get("sha", revision)
    siblings = metadata.get("siblings", [])
    parquet_paths = sorted(
        sibling["rfilename"]
        for sibling in siblings
        if sibling.get("rfilename", "").endswith(".parquet") and sibling["rfilename"].startswith(path_prefix)
    )
    if not parquet_paths:
        raise SystemExit(f"No parquet files found for {dataset_id} with prefix {path_prefix!r}.")

    download_dir = output_root / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_root / "raw.jsonl"
    written_docs = 0
    source_files: list[str] = []

    with raw_jsonl.open("w", encoding="utf-8") as handle:
        for parquet_path in parquet_paths:
            local_name = parquet_path.replace("/", "__")
            local_path = download_dir / local_name
            resolve_url = f"{endpoint}/datasets/{dataset_id}/resolve/{resolved_revision}/{parquet_path}"
            if not local_path.exists():
                with requests.get(resolve_url, stream=True, timeout=600, allow_redirects=True) as response:
                    response.raise_for_status()
                    with local_path.open("wb") as parquet_handle:
                        for chunk in response.iter_content(chunk_size=16 * 1024 * 1024):
                            if chunk:
                                parquet_handle.write(chunk)
            source_files.append(parquet_path)
            parquet_file = pq.ParquetFile(local_path)
            for batch in parquet_file.iter_batches(columns=["text"], batch_size=1024):
                texts = batch.column("text").to_pylist()
                for text in texts:
                    handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    written_docs += 1
                    if written_docs >= limit_docs:
                        return written_docs, {
                            "hf_dataset": dataset_id,
                            "hf_endpoint": endpoint,
                            "resolved_revision": resolved_revision,
                            "path_prefix": path_prefix,
                            "source_files": source_files,
                        }
    return written_docs, {
        "hf_dataset": dataset_id,
        "hf_endpoint": endpoint,
        "resolved_revision": resolved_revision,
        "path_prefix": path_prefix,
        "source_files": source_files,
    }


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
    data_path = Path(dataset_cfg["data_path"])
    output_prefix = preprocess_output_prefix(data_path)
    limit_docs = args.limit_docs or dataset_cfg.get("document_count", 50000)

    idx_path = Path(f"{data_path}.idx")
    bin_path = Path(f"{data_path}.bin")
    if idx_path.exists() and bin_path.exists() and not args.overwrite:
        manifest = {
            "dataset_spec_id": args.dataset_spec_id,
            "hf_dataset": dataset_cfg["hf_dataset"],
            "revision": dataset_cfg.get("revision"),
            "document_count": dataset_cfg.get("document_count", limit_docs),
            "raw_jsonl": str(raw_jsonl) if raw_jsonl.exists() else None,
            "data_path": str(data_path),
            "preprocess_output_prefix": str(output_prefix),
            "tokenizer_model": dataset_cfg.get("tokenizer_model"),
            "already_preprocessed": True,
        }
        (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return 0

    manifest_extra: dict[str, object] = {}
    if raw_jsonl.exists() and not args.overwrite:
        selected_count = dataset_cfg.get("document_count", limit_docs)
    elif dataset_cfg.get("hf_transport") == "mirror_api":
        selected_count, manifest_extra = load_fineweb_docs_via_mirror(dataset_cfg, limit_docs, output_root)
    else:
        ds = load_dataset(dataset_cfg["hf_dataset"], revision=dataset_cfg.get("revision"), split="train")
        selected = ds.select(range(min(limit_docs, len(ds))))
        with raw_jsonl.open("w", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
        selected_count = len(selected)

    tokenizer_model = dataset_cfg.get("tokenizer_model") or next(
        model.get("tokenizer_model")
        for model in matrix["models"].values()
        if model.get("tokenizer_model")
    )

    obs_python = os.environ.get("OBS_PYTHON", "python3")
    cmd = [
        obs_python,
        str(PREPROCESS_SCRIPT),
        "--input",
        str(raw_jsonl),
        "--output-prefix",
        str(output_prefix),
        "--tokenizer-type",
        dataset_cfg.get("tokenizer_type", "HuggingFaceTokenizer"),
        "--tokenizer-model",
        tokenizer_model,
        "--append-eod",
        "--workers",
        "1",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    manifest = {
        "dataset_spec_id": args.dataset_spec_id,
        "hf_dataset": dataset_cfg["hf_dataset"],
        "revision": dataset_cfg.get("revision"),
        "document_count": selected_count,
        "raw_jsonl": str(raw_jsonl),
        "data_path": str(data_path),
        "preprocess_output_prefix": str(output_prefix),
        "tokenizer_model": tokenizer_model,
    }
    manifest.update(manifest_extra)
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
