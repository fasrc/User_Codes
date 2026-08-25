#!/usr/bin/env python3

"""
pi_signal.py

Estimate pi using a Monte Carlo "dart throwing" experiment with:

    - periodic checkpointing
    - checkpoint/restart
    - exact RNG-state restoration
    - graceful checkpointing when a Unix signal is received

This example is designed for Slurm using SIGUSR1.

Example
-------

Start:

    python pi_signal.py \
        --darts 100000000 \
        --seed 42 \
        --report-every 1000000 \
        --checkpoint-every 5000000 \
        --sleep 1

Resume:

    python pi_signal.py \
        --darts 100000000 \
        --seed 42 \
        --report-every 1000000 \
        --checkpoint-every 5000000 \
        --sleep 1 \
        --resume
"""

import argparse
import math
import os
import pickle
import random
import signal
import sys
import time
from pathlib import Path


CHECKPOINT_VERSION = 1

# ----------------------------------------------------------------------
# Signal state
# ----------------------------------------------------------------------

stop_requested = False
received_signal = None


def signal_handler(signum, frame):
    """
    Minimal signal handler.

    Do NOT perform checkpoint I/O here.  We simply record that a signal
    was received.  The main computation will notice the flag and save a
    checkpoint at a safe point.
    """

    global stop_requested
    global received_signal

    stop_requested = True
    received_signal = signum


# ----------------------------------------------------------------------
# Command-line arguments
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate pi using Monte Carlo dart throwing with "
            "checkpoint/restart and signal handling."
        )
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
        help="Write a periodic checkpoint every N darts "
             "(default: 100,000)",
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
        help="Resume from an existing checkpoint",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay after each progress report "
             "(useful for demonstrations)",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Checkpoint functions
# ----------------------------------------------------------------------

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
    Save a checkpoint using temporary-file + atomic replacement.
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

    checkpoint_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_file = checkpoint_file.with_name(
        checkpoint_file.name + ".tmp"
    )

    with open(tmp_file, "wb") as f:
        pickle.dump(
            checkpoint,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, checkpoint_file)


def load_checkpoint(
    checkpoint_file,
    expected_darts,
    expected_seed,
):
    """
    Load and validate a checkpoint.
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


# ----------------------------------------------------------------------
# Monte Carlo calculation
# ----------------------------------------------------------------------

def estimate_pi(args):

    rng = random.Random(args.seed)

    completed_darts = 0
    inside_circle = 0
    previous_elapsed = 0.0

    #
    # Restore an earlier checkpoint if requested.
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

        rng.setstate(checkpoint["rng_state"])

        print()
        print("=" * 72)
        print("Restarting from checkpoint")
        print("=" * 72)
        print(f"Completed darts    : {completed_darts:,}")
        print(f"Inside circle      : {inside_circle:,}")
        print(f"Previous runtime   : {previous_elapsed:.2f} s")
        print("=" * 72)
        print()

    start_time = time.perf_counter()

    #
    # Main computation.
    #
    for i in range(completed_darts + 1, args.darts + 1):

        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)

        if x * x + y * y <= 1.0:
            inside_circle += 1

        #
        # --------------------------------------------------------------
        # Signal-triggered checkpoint
        # --------------------------------------------------------------
        #
        # Check the flag at a safe point after completing the current
        # Monte Carlo iteration.
        #
        if stop_requested:

            elapsed = (
                previous_elapsed
                + time.perf_counter()
                - start_time
            )

            print()
            print("=" * 72)
            print("Termination signal received")
            print("=" * 72)

            if received_signal is not None:
                print(
                    f"Signal             : "
                    f"{signal.Signals(received_signal).name}"
                )

            print(f"Completed darts    : {i:,}")
            print("Saving checkpoint ...")

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
                f"Checkpoint saved   : "
                f"{args.checkpoint_file}"
            )
            print("Exiting cleanly.")
            print("=" * 72)

            return None

        #
        # --------------------------------------------------------------
        # Progress report
        # --------------------------------------------------------------
        #
        if args.report_every > 0 and (
            i % args.report_every == 0
            or i == args.darts
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
                f"error = "
                f"{abs(pi_estimate - math.pi):.3e}   "
                f"elapsed = {elapsed:.2f} s",
                flush=True,
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

        #
        # --------------------------------------------------------------
        # Periodic checkpoint
        # --------------------------------------------------------------
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
                f"  --> periodic checkpoint saved "
                f"at dart {i:,}",
                flush=True,
            )

    #
    # Calculation completed normally.
    #
    total_elapsed = (
        previous_elapsed
        + time.perf_counter()
        - start_time
    )

    pi_estimate = (
        4.0 * inside_circle / args.darts
    )

    save_checkpoint(
        checkpoint_file=args.checkpoint_file,
        completed_darts=args.darts,
        inside_circle=inside_circle,
        rng=rng,
        n_darts=args.darts,
        seed=args.seed,
        elapsed=total_elapsed,
    )

    return (
        pi_estimate,
        inside_circle,
        total_elapsed,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    args = parse_args()

    if args.darts <= 0:
        raise ValueError(
            "--darts must be greater than zero"
        )

    if args.report_every < 0:
        raise ValueError(
            "--report-every cannot be negative"
        )

    if args.checkpoint_every < 0:
        raise ValueError(
            "--checkpoint-every cannot be negative"
        )

    if args.sleep < 0:
        raise ValueError(
            "--sleep cannot be negative"
        )

    #
    # Register signal handlers.
    #
    # SIGUSR1:
    #     Our preferred advance-warning signal from Slurm.
    #
    # SIGTERM:
    #     Also handled as a fallback termination signal.
    #
    signal.signal(
        signal.SIGUSR1,
        signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        signal_handler,
    )

    print("=" * 72)
    print(
        "Monte Carlo Pi Estimation "
        "with Signal-Aware Checkpointing"
    )
    print("=" * 72)
    print(f"Process ID           : {os.getpid()}")
    print(f"Number of darts      : {args.darts:,}")
    print(f"Random seed          : {args.seed}")
    print(f"Report interval      : {args.report_every:,}")
    print(
        f"Checkpoint interval  : "
        f"{args.checkpoint_every:,}"
    )
    print(
        f"Checkpoint file      : "
        f"{args.checkpoint_file}"
    )
    print(f"Resume               : {args.resume}")
    print(f"Sleep interval       : {args.sleep} s")
    print("=" * 72)

    result = estimate_pi(args)

    #
    # result == None means we checkpointed because a signal
    # requested graceful termination.
    #
    if result is None:
        sys.exit(0)

    pi_estimate, inside_circle, elapsed = result

    print()
    print("=" * 72)
    print("Final result")
    print("=" * 72)
    print(f"Darts thrown       : {args.darts:,}")
    print(f"Inside circle      : {inside_circle:,}")
    print(f"Estimated pi       : {pi_estimate:.10f}")
    print(f"Python math.pi     : {math.pi:.10f}")
    print(
        f"Absolute error     : "
        f"{abs(pi_estimate - math.pi):.6e}"
    )
    print(f"Total elapsed time : {elapsed:.2f} s")
    print("=" * 72)


if __name__ == "__main__":
    main()
