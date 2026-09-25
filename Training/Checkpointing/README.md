# Checkpointing on CANNON

Long-running jobs get interrupted: a wall-time limit ends them, a preemptible
partition takes the node back, or a node fails. **Checkpoint/restart** saves the
state of a computation to disk so a later job can continue from it instead of
starting over.

These exercises show how to checkpoint in six settings, from a hand-written Python
loop to a transparent MPI checkpoint and a GPU training run. Every exercise
follows the same pattern:

```
compute → checkpoint → compute → [interruption] → restart → load checkpoint → continue
```

---

## Exercises

| # | Exercise | Technique | Application | Runs on |
|---|----------|-----------|-------------|---------|
| 1 | [Checkpointing in Python](Exercise1/README.md) | Application-level | Monte Carlo π (Python) | CPU, interactive and batch |
| 2 | [Checkpointing in C](Exercise2/README.md) | Application-level | Monte Carlo π (C) | CPU, interactive and batch |
| 3 | [Transparent checkpointing with DMTCP](Exercise3/README.md) | Transparent, single process | Long-running C program | CPU, interactive and batch |
| 4 | [Transparent MPI checkpointing with MANA](Exercise4/README.md) | Transparent, MPI | Monte Carlo π on 4 MPI ranks | CPU, interactive and batch |
| 5 | [Checkpointing in GROMACS](Exercise5/README.md) | Built into the application | Molecular dynamics of water | CPU (MPI), batch |
| 6 | [Checkpointing in PyTorch](Exercise6/README.md) | Application-level | MNIST training | GPU, interactive and batch |

### Exercise 1 — Checkpointing in Python

A Monte Carlo estimate of π is made progressively more robust in four steps: no
checkpointing; periodic, atomic checkpoints with a manual `--resume`;
signal-aware checkpointing that reacts to Slurm's `SIGUSR1` time-limit warning;
and automatic restart after a requeue, where the program finds its checkpoint by
itself. The checkpoint holds the counters, the random-number-generator state and
the run parameters, written with `pickle`.

### Exercise 2 — Checkpointing in C

The same four steps in C, where the details Python hides become visible: an
explicit checkpoint structure written as a binary file, the `fflush` / `fsync` /
`rename` sequence for atomic writes, `sigaction` signal handlers that only set a
flag, and a small random-number generator whose whole state fits in one integer.

### Exercise 3 — Transparent checkpointing with DMTCP

Checkpointing without touching the program: DMTCP saves and restores a running
process from outside. A long-running C program with no checkpoint code is run
under DMTCP interactively, then in batch mode with periodic checkpoints and a
restart script, plus an optional automatic checkpoint/requeue/restart example.

### Exercise 4 — Transparent MPI checkpointing with MANA

MANA extends the transparent approach to MPI. An MPI program built with
`mpicc_mana` runs on four ranks under MANA, is checkpointed from a second shell
(one checkpoint directory per rank), stopped, and restarted, first interactively
and then with batch scripts. It covers the practical details: `srun --mpi=pmix`,
restarting from inside the checkpoint directory, and passing the Slurm job ID to
the second shell.

### Exercise 5 — Checkpointing in GROMACS

Many scientific codes checkpoint themselves. A GROMACS water simulation runs on
four MPI ranks with `mdrun -cpt` writing `md.cpt`, is stopped cleanly by `-maxh`,
and is restarted with `-cpi md.cpt`. This exercise covers the manual stop and
restart in batch mode.

### Exercise 6 — Checkpointing in PyTorch

What a training checkpoint has to contain, in three versions of an MNIST
classifier trained on a GPU: no checkpointing; a basic checkpoint (epoch, model
and optimizer); and a complete checkpoint that adds the learning-rate scheduler,
early-stopping state and all random-number generators. Each version is run
interactively and then in batch mode on the `gpu` partition.

---

## Choosing an approach

| Approach | Exercises | You get | You need |
|----------|-----------|---------|----------|
| Application-level | 1, 2, 6 | Small, portable checkpoints; you decide exactly what is saved | To change the code |
| Built into the application | 5 | Checkpointing that already works and is maintained for you | An application that provides it |
| Transparent | 3, 4 | No code changes at all | A supported environment (DMTCP needs dynamically linked programs; MANA needs the MPI program built with `mpicc_mana`) |

If the code is yours, application-level checkpointing is the most robust. If the
application already checkpoints, use that. Transparent tools are for code you
cannot or do not want to change.

---

## Suggested order

Exercises 1 and 2 introduce the concepts (what state to save, atomic writes,
signals, automatic restart) and are the best starting point; do Exercise 2 after
Exercise 1 to see the same ideas at a lower level. Exercises 3 and 4 then show the
transparent alternative for a single process and for MPI. Exercises 5 and 6 apply
the ideas to scientific and machine-learning software and can be done in any
order.

---

## How the exercises are organized

Each exercise directory has a `README.md` with the same layout:

- **Introduction** — what the exercise shows.
- **Content** — the files in the directory.
- **Workflow** — a setup step, then numbered steps with the exact commands and
  example output; the first steps are usually interactive, the later ones batch.
- **Cleanup** — how to remove generated files before running again.

The example programs include a `--sleep` option (or an equivalent) that slows the
run so there is time to interrupt it, and the batch scripts use short time limits
on purpose so the checkpoint and restart can be seen quickly.

---

## Prerequisites

- An account on CANNON with access to the partitions used: the test partitions
  (`test`, `rc-testing`), `serial_requeue` for the requeue examples, and `gpu` for
  Exercise 6.
- Software comes from environment modules and shared installations named in each
  exercise (GCC, DMTCP, MPICH, MANA, GROMACS, and a PyTorch conda environment).
