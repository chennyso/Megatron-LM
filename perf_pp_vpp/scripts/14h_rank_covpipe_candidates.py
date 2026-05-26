#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank COVPipe candidates with the lightweight estimator.")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--baseline-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compiler = load_module(Path(__file__).with_name("14_generate_covpipe_candidate.py"), "covpipe_compiler")
    estimator = load_module(Path(__file__).with_name("14g_estimate_covpipe_candidate.py"), "covpipe_estimator")
    baseline = json.loads(
        __import__("subprocess").check_output(
            [
                "python",
                str(Path(__file__).with_name("14b_score_covpipe_profile.py")),
                str(args.baseline_profile),
            ],
            text=True,
        )
    )

    rows = []
    for path in sorted(args.candidate_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        compiled = compiler.compile_candidate(json.loads(path.read_text(encoding="utf-8")))
        rows.append(estimator.estimate(compiled, baseline))

    rows.sort(
        key=lambda row: (
            row["estimated_exposed_comm_proxy_ns"],
            -row["estimated_covering_ratio"],
        )
    )
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
