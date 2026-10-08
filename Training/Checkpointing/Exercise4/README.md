# Exercise 4 — Transparent MPI Checkpointing with MANA

## Introduction

Exercise 3 used **DMTCP** to checkpoint a single process without changing the
program. This exercise extends the same idea to an **MPI application** with
[MANA](https://github.com/mpickpt/mana), which provides transparent
checkpoint/restart for distributed MPI computations. The pattern is:

```
launch under MANA → compute → checkpoint → [interruption] → restart → continue
```

The vehicle is again a Monte Carlo estimate of $\pi$: random points $(x, y)$ are
thrown into a unit square, and the fraction landing inside the unit circle
($x^2 + y^2 \le 1$) converges to $\pi / 4$:

$$
\pi \approx 4 \times \frac{\text{points inside circle}}{\text{total points}}
$$

The darts are distributed over four MPI ranks, each with its own random-number
generator, and rank 0 reports the combined estimate. The program `pi_mpi.c`
contains **no checkpoint/restart logic**.

The exercise has four steps: the native MPI program, checkpoint and restart under
MANA, and then the same checkpoint and restart in batch mode. The first two run
**interactively**, the last two in **batch** mode. Automatic Slurm requeue is not
part of this exercise.

Unlike Exercises 1 and 2, the program does not decide what to save. MANA captures
the state of all MPI ranks from outside and writes one checkpoint directory per
rank:

```
ckpt_rank_0/   ckpt_rank_1/   ckpt_rank_2/   ckpt_rank_3/
```

Key ideas along the way:

- **Transparent checkpointing** — MANA saves and restores the process and MPI
  state; the application does not know it is being checkpointed.
- **Coordinated checkpoint** — all ranks belong to one distributed computation and
  are checkpointed consistently, so a restart needs the same number of ranks.
- **Build and launch details matter** — the executable run under MANA must be
  built with `mpicc_mana`, and with this MPICH build Slurm needs
  `srun --mpi=pmix`.
- **Restart location matters** — with this MANA installation, restart from inside
  the checkpoint directory.

---

## Content

| File | Description | Mode |
|------|-------------|------|
| `pi_mpi.c` | Monte Carlo π estimate distributed over MPI ranks, no checkpoint logic | — |
| `Makefile` | Builds `pi_mpi.x` (with `mpicc`) and `pi_mpi_mana.x` (with `mpicc_mana`) | — |
| `setup.sh` | Loads GCC and MPICH and puts MANA on `PATH` (copy of the shared `setup_mana.sh`) | — |
| `mana_checkpoint.sbatch` | Launches the program under MANA with a checkpoint every 15 s | Batch |
| `mana_restart.sbatch` | Restarts the program from `checkpoints/` | Batch |
| `mana_launch_51392273.out` | Example output of `mana_checkpoint.sbatch` (ran until its time limit at 20.80%; last checkpoint just before the 19.20% report) | Batch |
| `mana_restart_51395336.out` | Example output of `mana_restart.sbatch` (resumed from that checkpoint) | Batch |

---

## Workflow

### Setup

Steps 1 and 2 run on a compute node, not a login node. Then load the environment,
build both executables and create the checkpoint directory:

```bash
salloc -n 4 -N 1 -t 180 -p test --mem-per-cpu=1G
source /n/holylabs/rc_admin/Everyone/checkpoint-training/setup_mana.sh
make
mkdir -p checkpoints
```
The `source` command also loads the required modules, i.e.,

```bash
module load gcc/15.2.0-fasrc01
module load mpich/5.0.0-fasrc01
```

`make` creates `pi_mpi.x` (standard MPI build, used for the native baseline) and
`pi_mpi_mana.x` (built with `mpicc_mana`, used with MANA). To check the
environment:

```bash
which mpicc mpicc_mana mana_coordinator mana_launch mana_status mana_restart
```

### 1. Native MPI — `pi_mpi.x` (interactive)

The baseline runs under Slurm with no checkpointing. With this MPICH build, use
`srun --mpi=pmix`.

1. Run the program on four ranks:

   ```bash
   srun --mpi=pmix -n 4 ./pi_mpi.x --darts 100000000
   ```

   ```
   ============================================================
   MPI Monte Carlo Pi
   ============================================================
   MPI ranks          : 4
   Total darts        : 100000000
   Base seed          : 12345
   Report every       : 1000000 local darts
   Sleep after report : 0 ms
   ============================================================
   Progress:   4.00%  darts=4000000  pi=3.1413600000  error=-2.327e-04  elapsed=0.01 s
   Progress:   8.00%  darts=8000000  pi=3.1410140000  error=-5.787e-04  elapsed=0.01 s
   Progress:  12.00%  darts=12000000  pi=3.1411943333  error=-3.983e-04  elapsed=0.02 s
   Progress:  16.00%  darts=16000000  pi=3.1413702500  error=-2.224e-04  elapsed=0.02 s
   ...
   ============================================================
   Finished
   ============================================================
   Darts       : 100000000
   Inside      : 78536313
   Pi estimate : 3.141452520000
   Pi true     : 3.141592653590
   Error       : -1.401336e-04
   Elapsed     : 0.128 s
   ============================================================
   ```

2. The progress lines end with the final estimate of π. Nothing is saved, so an
   interruption would lose everything.

### 2. Checkpoint and restart with MANA — `pi_mpi_mana.x` (interactive)

MANA is controlled through a coordinator. `mana_coordinator -i15` starts one that
also checkpoints automatically every 15 seconds. You use two shells on the same
compute node: shell 1 runs the calculation, shell 2 controls MANA.

1. **Shell 1** — note the job ID, then start the coordinator and launch the
   MANA-built program. The large dart count and `--sleep-ms 1000` keep it running
   long enough to interact with:

   ```bash
   echo $SLURM_JOB_ID
   mana_coordinator -i15

   srun --mpi=pmix -n 4 \
       mana_launch \
       --ckptdir checkpoints \
       ./pi_mpi_mana.x \
       --darts 1000000000 \
       --report-every 5000000 \
       --sleep-ms 1000
   ```

   ```
   ============================================================
   MPI Monte Carlo Pi
   ============================================================
   MPI ranks          : 4
   Total darts        : 1000000000
   Base seed          : 12345
   Report every       : 5000000 local darts
   Sleep after report : 1000 ms
   ============================================================
   Progress:   2.00%  darts=20000000  pi=3.1416424000  error=+4.975e-05  elapsed=0.03 s
   Progress:   4.00%  darts=40000000  pi=3.1413395000  error=-2.532e-04  elapsed=1.05 s
   ...
   ```

2. **Shell 2** — log in to the same compute node and set up the same
   environment. The second shell does not inherit the Slurm environment, so
   export the job ID from step 1 (replace `51384650` with yours; `squeue -u $USER`
   also shows it). MANA uses it to find its coordinator file:

   ```bash
   source /n/holylabs/rc_admin/Everyone/checkpoint-training/setup_mana.sh

   export SLURM_JOB_ID=51384650 # Use the actual JobID
   ls -l ~/.mana-slurm-${SLURM_JOB_ID}.rc
   mana_status --status
   ```

3. Request a coordinated checkpoint and inspect it:

   ```bash
   mana_status --checkpoint
   ls checkpoints
   ```

   ```
   ckpt_rank_0  ckpt_rank_1  ckpt_rank_2  ckpt_rank_3
   ```

   Each rank directory holds a small `header.mana` file and a checkpoint image of
   about 35 MB:

   ```bash
   ls checkpoints/ckpt_rank_0
   ```

   ```
   ckpt_lower-half_100242aeaf00ee78-41000-ab449521515b2.dmtcp  header.mana
   ```

4. Stop the running calculation. The `srun` command in shell 1 terminates and the
   checkpoint files stay in `checkpoints/`:

   ```bash
   mana_status --quit
   ```

5. **Shell 1** — restart from inside the checkpoint directory, with a new
   coordinator and the same number of ranks:

   ```bash
   cd checkpoints
   mana_coordinator -i15
   srun --mpi=pmix -n 4 mana_restart
   ```

   ```
   Progress:  24.00%  darts=240000000  pi=3.1416401500  error=+4.750e-05  elapsed=49.10 s
   Progress:  26.00%  darts=260000000  pi=3.1416156923  error=+2.304e-05  elapsed=50.13 s
   Progress:  28.00%  darts=280000000  pi=3.1416204429  error=+2.779e-05  elapsed=51.16 s
   Progress:  30.00%  darts=300000000  pi=3.1416058000  error=+1.315e-05  elapsed=52.18 s
   Progress:  32.00%  darts=320000000  pi=3.1416087250  error=+1.607e-05  elapsed=53.21 s
   Progress:  34.00%  darts=340000000  pi=3.1415853765  error=-7.277e-06  elapsed=54.23 s
   ...
   ``` 

   The progress lines continue from about where the checkpoint was taken instead
   of starting again from zero. Work done after the last checkpoint is repeated.

### 3. Checkpoint in batch mode — `mana_checkpoint.sbatch` (batch)

The script starts a coordinator with `-i15`, launches `pi_mpi_mana.x` under MANA on
four ranks and checkpoints every 15 seconds. It removes any existing `checkpoints/`
directory first. The calculation (10 billion darts at `--sleep-ms 1000`) is
deliberately longer than the 2-minute time limit, so the job ends with state
`TIMEOUT` and the last periodic checkpoint remains.

1. Submit and follow the job (replace `JOBID` with the job ID printed by
   `sbatch`):

   ```bash
   sbatch mana_checkpoint.sbatch
   squeue -u $USER
   tail -f mana_launch_JOBID.out
   ```

   ```
   *** Coordinator/job information written to /n/home06/pkrastev/.mana-slurm-15390.rc
   ============================================================
   MANA checkpoint job
   ============================================================
   Job ID : 15390
   ...
   Progress:   0.20%  darts=20000000  pi=3.1416424000  error=+4.975e-05  elapsed=0.03 s
   Progress:   0.40%  darts=40000000  pi=3.1413395000  error=-2.532e-04  elapsed=1.05 s
   ...
   ```

   The occasional jump in `elapsed` (for example from 118.67 s to 125.29 s) is the
   pause while MANA writes a checkpoint.

2. After the job has ended, inspect the checkpoint files:

   ```bash
   find checkpoints -maxdepth 2 -type f -ls
   ```

   There is one directory per rank, each with a `header.mana` file and a
   `ckpt_lower-half_*.dmtcp` image.

### 4. Restart in batch mode — `mana_restart.sbatch` (batch)

The restart script sources the same environment and checks that `checkpoints/`
exists and contains one `ckpt_rank_*` directory per MPI task. It then changes into
that directory, starts a new coordinator and runs `mana_restart` on four ranks.

1. Submit and follow the job:

   ```bash
   sbatch mana_restart.sbatch
   tail -f mana_restart_JOBID.out
   ```

   ```
   ============================================================
   MANA restart
   ============================================================
   Job ID          : 15392
   ...
   Found 4 checkpoint rank directories.
   ...
   Restarting MPI application...
   Progress:  19.20%  darts=1920000000  pi=3.1416034917  error=+1.084e-05  elapsed=687.78 s
   Progress:  19.40%  darts=1940000000  pi=3.1416068660  error=+1.421e-05  elapsed=688.81 s
   ```

2. The calculation resumes at 19.20% instead of 0%. The checkpointed job had
   reached 20.80%, but its last checkpoint was taken just before the 19.20%
   report, so that part is computed again. The `elapsed` value keeps counting from
   the original start, so it includes the time between the checkpoint and the
   restart.

### Cleanup

Leave the allocation, then remove generated files:

```bash
exit
rm -rf checkpoints
rm -f mana_launch_*.out mana_launch_*.err mana_restart_*.out mana_restart_*.err
make clean
```

---

## References

- MANA: <https://github.com/mpickpt/mana>
- MANA documentation: <https://mana-doc.readthedocs.io/>
- NERSC MANA documentation: <https://docs.nersc.gov/development/checkpoint-restart/mana/>
