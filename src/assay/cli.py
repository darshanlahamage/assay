"""Assay CLI.

    assay run    --task <dir>         run the agent on one task (one episode)
    assay train  --suite <dir> -n N   run the self-improvement loop for N iters
    assay eval   --ledger <path>      build the results table + plots

Phase-1 scaffold: argument parsing + wiring are in place; the heavy steps raise
NotImplementedError where the corresponding module is still a stub.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assay", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the agent on one task")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--config", default="configs/default.yaml")

    p_train = sub.add_parser("train", help="run the self-improvement loop")
    p_train.add_argument("--suite", required=True)
    p_train.add_argument("-n", "--iterations", type=int, default=5)
    p_train.add_argument("--config", default="configs/default.yaml")

    p_eval = sub.add_parser("eval", help="build the results table + plots")
    p_eval.add_argument("--ledger", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from assay.pipeline import run_task          # noqa: PLC0415
        return run_task(args.task, args.config)
    if args.cmd == "train":
        from assay.pipeline import train_loop         # noqa: PLC0415
        return train_loop(args.suite, args.iterations, args.config)
    if args.cmd == "eval":
        from evals.run_eval import build_report        # noqa: PLC0415
        return build_report(args.ledger)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
