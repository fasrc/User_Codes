#!/usr/bin/env python3

"""
pi_naive.py

Estimate pi using a Monte Carlo "dart throwing" experiment.

This is the baseline version for the checkpointing workshop.
It intentionally contains NO checkpoint/restart functionality.

Example:
    python pi_naive.py --darts 1000000 --seed 42
"""

import argparse
import math
import random
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate pi using Monte Carlo dart throwing."
    )

    parser.add_argument(
        "-n",
        "--darts",
        type=int,
        default=1_000_000,
        help="Number of darts to throw (default: 1,000,000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random-number generator seed (default: 42)",
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=100_000,
        help="Print progress every N darts (default: 100,000)",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help=(
            "Optional delay in seconds after each progress interval. "
            "Useful for checkpoint/restart demonstrations (default: 0)"
        ),
    )

    return parser.parse_args()


def estimate_pi(n_darts, seed, report_every, sleep_time):
    """
    Estimate pi by randomly throwing points into the square [-1, 1] x [-1, 1].

    The fraction of points falling inside the unit circle approaches pi/4.
    """

    rng = random.Random(seed)

    inside_circle = 0

    start_time = time.perf_counter()

    for i in range(1, n_darts + 1):

        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)

        if x * x + y * y <= 1.0:
            inside_circle += 1

        if report_every > 0 and (
            i % report_every == 0 or i == n_darts
        ):
            pi_estimate = 4.0 * inside_circle / i
            elapsed = time.perf_counter() - start_time

            print(
                f"darts = {i:>12,d} / {n_darts:,}   "
                f"pi = {pi_estimate:.8f}   "
                f"error = {abs(pi_estimate - math.pi):.3e}   "
                f"elapsed = {elapsed:.2f} s",
                flush=True,
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

    elapsed = time.perf_counter() - start_time

    pi_estimate = 4.0 * inside_circle / n_darts

    return pi_estimate, inside_circle, elapsed


def main():
    args = parse_args()

    if args.darts <= 0:
        raise ValueError("--darts must be greater than zero")

    if args.report_every < 0:
        raise ValueError("--report-every cannot be negative")

    if args.sleep < 0:
        raise ValueError("--sleep cannot be negative")

    print("=" * 72)
    print("Monte Carlo Pi Estimation")
    print("=" * 72)
    print(f"Number of darts : {args.darts:,}")
    print(f"Random seed     : {args.seed}")
    print(f"Report interval : {args.report_every:,}")
    print(f"Sleep interval  : {args.sleep} s")
    print("=" * 72)

    pi_estimate, inside_circle, elapsed = estimate_pi(
        n_darts=args.darts,
        seed=args.seed,
        report_every=args.report_every,
        sleep_time=args.sleep,
    )

    print()
    print("=" * 72)
    print("Final result")
    print("=" * 72)
    print(f"Darts thrown       : {args.darts:,}")
    print(f"Inside circle      : {inside_circle:,}")
    print(f"Estimated pi       : {pi_estimate:.10f}")
    print(f"Python math.pi     : {math.pi:.10f}")
    print(f"Absolute error     : {abs(pi_estimate - math.pi):.6e}")
    print(f"Total elapsed time : {elapsed:.2f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()
