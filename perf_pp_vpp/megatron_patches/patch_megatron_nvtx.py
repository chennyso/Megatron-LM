#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "import perf_pp_vpp.megatron_patches.nvtx_instrumentation  # perf_pp_vpp bootstrap\n"


def patch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        return
    path.write_text(MARKER + text, encoding="utf-8")


def unpatch(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace(MARKER, ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--megatron-root", required=True)
    parser.add_argument("--mode", choices=["apply", "revert", "smoke"], default="apply")
    args = parser.parse_args()
    target = Path(args.megatron_root) / "pretrain_gpt.py"
    if args.mode == "apply":
        patch(target)
    elif args.mode == "revert":
        unpatch(target)
    else:
        patch(target)
        import importlib.util
        spec = importlib.util.spec_from_file_location("pretrain_gpt", target)
        if spec is None:
            raise SystemExit("failed to load pretrain_gpt.py")
        print("smoke_ok")


if __name__ == "__main__":
    main()
