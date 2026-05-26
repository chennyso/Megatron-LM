#!/usr/bin/env python
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("binary_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    cmd = [args.binary]
    if args.binary_args and args.binary_args[0] == "--":
        cmd.extend(args.binary_args[1:])
    else:
        cmd.extend(args.binary_args)
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
