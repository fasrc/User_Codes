#!/usr/bin/env python3

"""
pi_requeue.py

Monte Carlo estimation of pi designed for Slurm requeue partitions.

Features
--------
- periodic application-level checkpoints
- automatic checkpoint discovery and restart
- exact RNG-state restoration
- atomic checkpoint writes
- SIGUSR1 and SIGTERM handling
- awareness of Slurm job/restart information

Unlike pi_checkpoint.py and pi_signal.py, no --resume option is needed.
If a checkpoint exists, the program automatically resumes from it.

Use --fresh to deliberately start over.
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

stop_requested = False
received_signal = None


# ----------------------------------------------------------------------
# Signal handling
# ----------------------------------------------------------------------

def signal_handler(signum, frame):
    """
    Keep the signal handler minimal.

    The handler only records that termination was requested.
    Checkpoint I/O is performed later from the normal program flow.
    """
    global stop_requested
    global received_signal

    stop_requested = True
    received_signal = signum


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Monte Carlo pi calculation with automatic "
            "checkpoint/restart for Slurm requeue jobs."
        )
    )

    parser.add_argument(
        "-n",
        "--darts",
        type=int,
        default=100_000_000,
        help="Total number of darts",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random-number generator seed",
    )

    parser.add_argument(
        "--report-every",
        type=int,
        default=1_000_000,
        help="Print progress every N darts",
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5_000_000,
        help="Write a periodic checkpoint every N darts",
    )

    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path("pi_checkpoint.pkl"),
        help="Checkpoint filename",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay after progress reports",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and remove any existing checkpoint",
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

    os.replace(
        tmp_file,
        checkpoint_file,
    )


def load_checkpoint(
    checkpoint_file,
    expected_darts,
    expected_seed,
):

    with open(checkpoint_file, "rb") as f:
        checkpoint = pickle.load(f)

    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise RuntimeError(
            "Checkpoint version is incompatible."
        )

    if checkpoint["n_darts"] != expected_darts:
        raise RuntimeError(
            "Checkpoint belongs to a different calculation:\n"
            f"  checkpoint darts : {checkpoint['n_darts']:,}\n"
            f"  requested darts  : {expected_darts:,}"
        )

    if checkpoint["seed"] != expected_seed:
        raise RuntimeError(
            "Checkpoint uses a different RNG seed:\n"
            f"  checkpoint seed : {checkpoint['seed']}\n"
            f"  requested seed  : {expected_seed}"
        )

    return checkpoint


# ----------------------------------------------------------------------
# Calculation
# ----------------------------------------------------------------------

def estimate_pi(args):

    rng = random.Random(args.seed)

    completed_darts = 0
    inside_circle = 0
    previous_elapsed = 0.0

    #
    # Automatic restart
    #
    if args.checkpoint_file.exists():

        checkpoint = load_checkpoint(
            checkpoint_file=args.checkpoint_file,
            expected_darts=args.darts,
            expected_seed=args.seed,
        )

        completed_darts = checkpoint["completed_darts"]
        inside_circle = checkpoint["inside_circle"]
        previous_elapsed = checkpoint["elapsed"]

        rng.setstate(
            checkpoint["rng_state"]
        )

        print()
        print("=" * 72)
        print("CHECKPOINT FOUND - AUTOMATIC RESTART")
        print("=" * 72)
        print(
            f"Completed darts  : "
            f"{completed_darts:,}"
        )
        print(
            f"Inside circle    : "
            f"{inside_circle:,}"
        )
        print(
            f"Previous runtime : "
            f"{previous_elapsed:.2f} s"
        )
        print("=" * 72)
        print()

        #
        # It is possible that a completed calculation
        # is being submitted again.
        #
        if completed_darts >= args.darts:

            pi_estimate = (
                4.0
                * inside_circle
                / completed_darts
            )

            return (
                pi_estimate,
                inside_circle,
                previous_elapsed,
                True,
            )

    else:

        print()
        print("No checkpoint found.")
        print("Starting a new calculation.")
        print()

    start_time = time.perf_counter()

    #
    # Main Monte Carlo loop
    #
    for i in range(
        completed_darts + 1,
        args.darts + 1,
    ):

        x = rng.uniform(-1.0, 1.0)
        y = rng.uniform(-1.0, 1.0)

        if x * x + y * y <= 1.0:
            inside_circle += 1

        #
        # Signal-triggered checkpoint
        #
        if stop_requested:

            elapsed = (
                previous_elapsed
                + time.perf_counter()
                - start_time
            )

            signal_name = signal.Signals(
                received_signal
            ).name

            print()
            print("=" * 72)
            print("SIGNAL RECEIVED")
            print("=" * 72)
            print(
                f"Signal           : {signal_name}"
            )
            print(
                f"Completed darts  : {i:,}"
            )
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
                f"Checkpoint saved : "
                f"{args.checkpoint_file}"
            )

            print("=" * 72)

            return None

        #
        # Progress report
        #
        if (
            args.report_every > 0
            and (
                i % args.report_every == 0
                or i == args.darts
            )
        ):

            pi_estimate = (
                4.0 * inside_circle / i
            )

            elapsed = (
                previous_elapsed
                + time.perf_counter()
                - start_time
            )

            print(
                f"darts = {i:>12,d} / "
                f"{args.darts:,}   "
                f"pi = {pi_estimate:.8f}   "
                f"error = "
                f"{abs(pi_estimate - math.pi):.3e}   "
                f"elapsed = {elapsed:.2f} s",
                flush=True,
            )

            if args.sleep > 0:
                time.sleep(args.sleep)

        #
        # Periodic checkpoint
        #
        if (
            args.checkpoint_every > 0
            and i % args.checkpoint_every == 0
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
                f"  --> periodic checkpoint "
                f"saved at dart {i:,}",
                flush=True,
            )

    #
    # Normal completion
    #
    total_elapsed = (
        previous_elapsed
        + time.perf_counter()
        - start_time
    )

    pi_estimate = (
        4.0
        * inside_circle
        / args.darts
    )

    #
    # Save the completed state.
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

    return (
        pi_estimate,
        inside_circle,
        total_elapsed,
        False,
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
    # Remove previous state only when explicitly requested.
    #
    if args.fresh and args.checkpoint_file.exists():

        print(
            f"Removing existing checkpoint: "
            f"{args.checkpoint_file}"
        )

        args.checkpoint_file.unlink()

    #
    # Register signals.
    #
    signal.signal(
        signal.SIGUSR1,
        signal_handler,
    )

    signal.signal(
        signal.SIGTERM,
        signal_handler,
    )

    #
    # Slurm information
    #
    job_id = os.environ.get(
        "SLURM_JOB_ID",
        "not-running-under-slurm",
    )

    restart_count = os.environ.get(
        "SLURM_RESTART_COUNT",
        "0",
    )

    hostname = os.environ.get(
        "HOSTNAME",
        "unknown",
    )

    print("=" * 72)
    print(
        "Monte Carlo Pi - "
        "Automatic Slurm Checkpoint/Restart"
    )
    print("=" * 72)

    print(f"Slurm job ID        : {job_id}")
    print(f"Slurm restart count : {restart_count}")
    print(f"Hostname            : {hostname}")

    print("-" * 72)

    print(
        f"Number of darts     : "
        f"{args.darts:,}"
    )

    print(
        f"Random seed         : "
        f"{args.seed}"
    )

    print(
        f"Checkpoint interval : "
        f"{args.checkpoint_every:,}"
    )

    print(
        f"Checkpoint file     : "
        f"{args.checkpoint_file}"
    )

    print("=" * 72)

    result = estimate_pi(args)

    #
    # Signal-triggered exit.
    #
    if result is None:

        print()
        print(
            "Application checkpointed "
            "and is terminating."
        )

        sys.exit(0)

    (
        pi_estimate,
        inside_circle,
        elapsed,
        already_complete,
    ) = result

    print()
    print("=" * 72)

    if already_complete:
        print(
            "Calculation was already complete."
        )
    else:
        print("CALCULATION COMPLETE")

    print("=" * 72)

    print(
        f"Darts thrown       : "
        f"{args.darts:,}"
    )

    print(
        f"Inside circle      : "
        f"{inside_circle:,}"
    )

    print(
        f"Estimated pi       : "
        f"{pi_estimate:.10f}"
    )

    print(
        f"Python math.pi     : "
        f"{math.pi:.10f}"
    )

    print(
        f"Absolute error     : "
        f"{abs(pi_estimate - math.pi):.6e}"
    )

    print(
        f"Total elapsed time : "
        f"{elapsed:.2f} s"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
