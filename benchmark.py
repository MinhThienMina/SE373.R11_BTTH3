"""Evaluation entry point for the three agent patterns."""

import argparse
import json
import sys

from flight_agent import evaluate, per_scenario, summarize


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="number of seeds per scenario")
    args = parser.parse_args()

    on = evaluate(args.n, harness=True)
    off = evaluate(args.n, harness=False)
    summarize(on, f"HARNESS ON ({len(on)} runs)")
    per_scenario(on)
    summarize(off, f"ABLATION - HARNESS OFF ({len(off)} runs)")
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump({"harness_on": on, "harness_off": off}, f, ensure_ascii=False, indent=1, default=str)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
