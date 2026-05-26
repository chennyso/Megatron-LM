#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", required=True)
    parser.add_argument("--profile-mode", choices=["none", "nsys"], default="none")
    parser.add_argument("--profile-ranks", default="none")
    parser.add_argument("--nsys-bin", default=os.environ.get("NSYS_BIN", "nsys"))
    parser.add_argument("--nsys-output-prefix", default="")
    parser.add_argument("--nsys-trace", default="cuda,nvtx,nccl,osrt")
    parser.add_argument("--patch-bootstrap", default="")
    parser.add_argument("entry_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def rank_selected(spec: str, rank: int) -> bool:
    spec = (spec or "none").strip()
    if spec == "all":
        return True
    if spec in ("none", ""):
        return False
    ranks = {int(x) for x in spec.split(",") if x.strip()}
    return rank in ranks


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    entry_args = list(args.entry_args)
    if entry_args and entry_args[0] == "--":
        entry_args = entry_args[1:]
    command = [sys.executable]
    if args.patch_bootstrap:
        command.extend(["-c", f"import runpy, sys; import {args.patch_bootstrap}; sys.argv=[{args.entry!r}] + {entry_args!r}; runpy.run_path({args.entry!r}, run_name='__main__')"])
    else:
        command.extend([args.entry, *entry_args])

    if args.profile_mode == "nsys" and rank_selected(args.profile_ranks, rank):
        out = f"{args.nsys_output_prefix}_rank{rank}"
        command = [
            args.nsys_bin,
            "profile",
            "--trace=" + args.nsys_trace,
            "--sample=none",
            "--cpuctxsw=none",
            "--cuda-memory-usage=true",
            "--force-overwrite=true",
            "-o",
            out,
            *command,
        ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
