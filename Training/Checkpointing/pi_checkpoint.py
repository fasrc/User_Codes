#!/usr/bin/env python3

"""
pi_checkpoint.py

Estimate pi using a Monte Carlo "dart throwing" experiment with
application-level checkpoint/restart.

The checkpoint contains all state required to continue the calculation:

    - number of completed darts
    - number of darts inside the unit circle
    - random-number generator state
    - elapsed runtime
    - run configuration

Examples
--------
Start a new calculation:

    python pi_checkpoint.py \
        --darts 100000000 \
        --seed 42 \
        --report-every 1000000 \
        --checkpoint-every 1000000 \
        --sleep 1

Restart from a checkpoint:

    python pi_checkpoint.py \
        --darts 100000000 \
        --seed 42 \
        --report-every 1000000 \
        --checkpoint-every 1000000 \
        --sleep 1 \
        --resume
"""

import argparse
import math
import os
import pickle
import random
import time
from pathlib import Path


CHECKPOINT_VERSION = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate pi using Monte Carlo dart throwing "
                    "with checkpoint/restart."
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
        "--checkpoint-every",
        type=int,
        default=100_000,
        help="Write a checkpoint every N darts (default: 100,000)",
    )

    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path("pi_checkpoint.pkl"),
        help="Checkpoint filename (default: pi_checkpoint.pkl)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume calculation from an existing checkpoint",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay after each progress report "
             "(useful for demonstrations)",
    )

    return parser.parse_args()


def save_checkpoint(
    checkpoint_file,
    completed_darts,
    inside_circle,
    rng,
    n_darts,
    seed,
    elapsed,
):
    """
    Save checkpoint atomically.

    The new checkpoint is first written to a temporary file. Only after
    the write succeeds is it atomically renamed to the real checkpoint
    filename.
    """

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "completed_darts": completed_darts,
        "inside_circle": inside_circle,
        "rng_state": rng.getstate(),
        "n_darts": n_darts,
        "seed": seed,
        "elapsed": elapsed,
    }

    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = checkpoint_file.with_name(
        checkpoint_file.name + ".tmp"
    )

    with open(tmp_file, "wb") as f:
        pickle.dump(
            checkpoint,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        # Push Python's buffered data to the operating system.
        f.flush()

        # Request that the operating system flush the file to storage.
        os.fsync(f.fileno())

    # Atomic replacement on the same filesystem.
    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(checkpoint_file, expected_darts, expected_seed):
    """
    Load and validate an existing checkpoint.
    """

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint file '{checkpoint_file}' does not exist."
        )

    with open(checkpoint_file, "rb") as f:
        checkpoint = pickle.load(f)

    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "Checkpoint version is incompatible with this program."
        )

    if checkpoint["n_darts"] != expected_darts:
        raise RuntimeError(
            "Checkpoint was created for a different number of darts:\n"
            f"  checkpoint : {checkpoint['n_darts']:,}\n"
            f"  requested  : {expected_darts:,}"
        )

    if checkpoint["seed"] != expected_seed:
        raise RuntimeError(
            "Checkpoint was created with a different random seed:\n"
            f"  checkpoint : {checkpoint['seed']}\n"
            f"  requested  : {expected_seed}"
        )

    return checkpoint


def estimate_pi(args):
    """
    Run or resume the Monte Carlo calculation.
    """

    rng = random.Random(args.seed)

    completed_darts = 0
    inside_circle = 0
    previous_elapsed = 0.0

    #
    # Restart if requested.
    #
    if args.resume:
        checkpoint = load_checkpoint(
            checkpoint_file=args.checkpoint_file,
            expected_darts=args.darts,
            expected_seed=args.seed,
        )

        completed_darts = checkpoint["completed_darts"]
        inside_circle = checkpoint["inside_circle"]
        previous_elapsed = checkpoint["elapsed"]

        # Restore the exact state of the random-number generator.
        rng.setstate(checkpoint["rng_state"])

        print()
        print("Checkpoint found.")
        print(f"Resuming from dart : {completed_darts:,}")
        print(f"Inside circle      : {inside_circle:,}")
        print(f"Previous runtime   : {previous_elapsed:.2f} s")
        print()

    start_time = time.perf_counter()

    #
    # Continue from the first dart not already completed.
    #
    for i in range(completed_darts + 1, args.darts + 1):

        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)

        if x * x + y * y <= 1.0:
            inside_circle += 1

        #
        # Progress report
        #
        if args.report_every > 0 and (
            i % args.report_every == 0 or i == args.darts
        ):
            pi_estimate = 4.0 * inside_circle / i

            elapsed = (
                previous_elapsed
                + time.perf_counter()
                - start_time
            )

            print(
                f"darts = {i:>12,d} / {args.darts:,}   "
                f"pi = {pi_estimate:.8f}   "
                f"error = {abs(pi_estimate - math.pi):.3e}   "
                f"elapsed = {elapsed:.2f} s",
                flush=True,
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

        #
        # Periodic checkpoint
        #
        if args.checkpoint_every > 0 and (
            i % args.checkpoint_every == 0
        ):
            elapsed = (
                previous_elapsed
                + time.perf_counter()
                - start_time
            )

            save_checkpoint(
                checkpoint_file=args.checkpoint_file,
                completed_darts=i,
                inside_circle=inside_circle,
                rng=rng,
                n_darts=args.darts,
                seed=args.seed,
                elapsed=elapsed,
            )

            print(
                f"  --> checkpoint saved at dart {i:,}",
                flush=True,
            )

    total_elapsed = (
        previous_elapsed
        + time.perf_counter()
        - start_time
    )

    pi_estimate = 4.0 * inside_circle / args.darts

    #
    # Write one final checkpoint representing the completed calculation.
    #
    save_checkpoint(
        checkpoint_file=args.checkpoint_file,
        completed_darts=args.darts,
        inside_circle=inside_circle,
        rng=rng,
        n_darts=args.darts,
        seed=args.seed,
        elapsed=total_elapsed,
    )

    return pi_estimate, inside_circle, total_elapsed


def main():
    args = parse_args()

    if args.darts <= 0:
        raise ValueError("--darts must be greater than zero")

    if args.report_every < 0:
        raise ValueError("--report-every cannot be negative")

    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every cannot be negative")

    if args.sleep < 0:
        raise ValueError("--sleep cannot be negative")

    print("=" * 72)
    print("Monte Carlo Pi Estimation with Checkpoint/Restart")
    print("=" * 72)
    print(f"Number of darts      : {args.darts:,}")
    print(f"Random seed          : {args.seed}")
    print(f"Report interval      : {args.report_every:,}")
    print(f"Checkpoint interval  : {args.checkpoint_every:,}")
    print(f"Checkpoint file      : {args.checkpoint_file}")
    print(f"Resume               : {args.resume}")
    print(f"Sleep interval       : {args.sleep} s")
    print("=" * 72)

    pi_estimate, inside_circle, elapsed = estimate_pi(args)

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

