# Checkpointing on CANNON

Long-running jobs get interrupted: a wall-time limit ends them, a preemptible
partition takes the node back, or a node fails. **Checkpoint/restart** saves the
state of a computation to disk so a later job can continue from it instead of
starting over. Every exercise follows the same pattern:

```
compute → checkpoint → compute → [interruption] → restart → load checkpoint → continue
```

---

## Exercises

| # | Exercise | Technique | What it covers | Runs on |
|---|----------|-----------|----------------|---------|
| 1 | [Checkpointing in Python](Exercise1/README.md) | Application-level | Monte Carlo π in four steps: no checkpoint, atomic checkpoints with `--resume`, reacting to Slurm's `SIGUSR1` warning, automatic restart after requeue | CPU, interactive and batch |
| 2 | [Checkpointing in C](Exercise2/README.md) | Application-level | The same four steps in C: a binary checkpoint structure, `fflush`/`fsync`/`rename` for atomic writes, `sigaction` handlers | CPU, interactive and batch |
| 3 | [Transparent checkpointing with DMTCP](Exercise3/README.md) | Transparent, single process | An unmodified C program checkpointed from outside, with periodic checkpoints, restart and automatic requeue | CPU, interactive and batch |
| 4 | [Transparent MPI checkpointing with MANA](Exercise4/README.md) | Transparent, MPI | A 4-rank MPI program built with `mpicc_mana`, checkpointed, stopped and restarted | CPU, interactive and batch |
| 5 | [Checkpointing in GROMACS](Exercise5/README.md) | Built into the application | A water simulation checkpointed with `mdrun -cpt`, stopped by `-maxh`, restarted with `-cpi` | CPU (MPI), batch |
| 6 | [Checkpointing in PyTorch](Exercise6/README.md) | Application-level | MNIST training with no, basic (model and optimizer) and complete (plus scheduler, early stopping, RNG) checkpoints on the `gpu_test` partition | GPU, interactive and batch |

**Suggested order:** start with Exercise 1, then 2 to see the same ideas at a
lower level. Exercises 3 and 4 show the transparent alternative; 5 and 6 can be
done in any order.

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

## How the exercises are organized

Each exercise `README.md` has the same sections: **Introduction**, **Content**
(the files), **Workflow** (setup, then numbered steps with commands and example
output, usually interactive first and batch after) and **Cleanup**.

The programs have a `--sleep` option (or equivalent) that slows them down so
there is time to interrupt them, and the batch scripts use deliberately short
time limits so checkpoint and restart can be seen quickly.
