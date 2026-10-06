"""CLI for the split-folder flight booking agent project."""

import argparse
import sys

from flight_agent import PATTERNS, demo, selftest


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["test", "demo"], help="run harness tests or one demo episode")
    parser.add_argument("pattern", nargs="?", default="hybrid", choices=list(PATTERNS))
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    if args.cmd == "test":
        selftest()
    else:
        demo(args.pattern, args.scenario, args.seed)


if __name__ == "__main__":
    main()
