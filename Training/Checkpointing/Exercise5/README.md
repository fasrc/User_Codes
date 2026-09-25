# Exercise 5 — Checkpointing in GROMACS

## Introduction

Many scientific applications have checkpoint/restart built in. This exercise uses
[GROMACS](https://www.gromacs.org/), a molecular dynamics package, to show the
same pattern as Exercises 1 and 2 with checkpointing that the application already
provides:

```
run → checkpoint → [stop] → restart from checkpoint → continue
```

The system is a box of 2,700 atoms (900 water molecules) simulated for 5,000,000
steps (5 ns) on four MPI ranks. `mdrun` writes its full simulation state to a
checkpoint file, `md.cpt`, at regular intervals (`-cpt`, in minutes) and again
when it stops. A later `mdrun -cpi md.cpt` reads that file and continues the
trajectory from the saved step instead of starting over. The checkpoint holds the
coordinates, velocities, step number and time, and the integrator state, so no
changes to the input are needed.

This exercise covers the **manual** workflow: a first job runs and stops, and a
second job that you submit yourself restarts it. Both run in **batch** mode.

Key ideas along the way:

- **Built-in checkpointing** — `-cpt N` sets the checkpoint interval in minutes,
  and `-cpi md.cpt` restarts from a checkpoint.
- **Clean stop before the time limit** — `-maxh H` makes `mdrun` finish after `H`
  hours, writing a final checkpoint, instead of being killed by the scheduler.
- **Same run, continued files** — a restart uses the same `-deffnm md` and appends
  to the existing `md.log` and `md.edr`.

---

## Content

| File | Description | Mode |
|------|-------------|------|
| `system.gro`, `system.top`, `system.mdp` | Input structure, topology and run parameters (GROMACS `int_water_md-vv_verlet_settle_pme_pme` test system, `nsteps = 5000000`) | — |
| `md.tpr` | Run input file generated with `grompp` from the three files above | — |
| `mdout.mdp` | Full parameter listing written by `grompp` | — |
| `gromacs_setup.sh` | Loads GCC and OpenMPI and adds the shared GROMACS 2026.3 install to the paths | — |
| `gromacs_checkpoint.sbatch` | Starts the simulation and stops it cleanly after 90 s, leaving `md.cpt` | Batch |
| `gromacs_restart.sbatch` | Restarts the simulation from `md.cpt` | Batch |
| `gmx_checkpoint_15254.out`, `gmx_checkpoint_15254.err` | Example output of `gromacs_checkpoint.sbatch` (stopped at step 24,390) | Batch |
| `gmx_restart_15255.out`, `gmx_restart_15255.err` | Example output of `gromacs_restart.sbatch` (continued from step 24,400) | Batch |

The GROMACS messages go to the `.err` files; the `.out` files hold the scripts' own
messages.

---

## Workflow

### Setup

Load the GROMACS environment and check that the MPI build is found:

```bash
source gromacs_setup.sh
which gmx_mpi
```

`md.tpr` is provided. Regenerate it only if you change `system.mdp`, for example
`nsteps`:

```bash
gmx_mpi grompp -f system.mdp -c system.gro -p system.top -o md.tpr
```

The run is intentionally much longer than the demonstration: 5,000,000 steps would
need about 200 slices of the length used here. The exercise shows one stop and one
restart.

### 1. Run and checkpoint — `gromacs_checkpoint.sbatch` (batch)

The script runs `mdrun` on four MPI ranks with two options that matter here:

```bash
gmx_mpi mdrun -deffnm md -cpt 0.1 -maxh 0.025
```

`-cpt 0.1` writes a checkpoint every 0.1 minutes. `-maxh 0.025` (90 seconds) makes
`mdrun` stop by itself before the job's 2-minute limit, as it would need to on a
real wall-time limit. The script refuses to start if `md.cpt` already exists, so an
old checkpoint is never continued by accident.

1. Make sure no old checkpoint or output is present, then submit and follow the
   job (replace `JOBID` with the job ID printed by `sbatch`):

   ```bash
   rm -f md.cpt md_prev.cpt md.edr md.log md.gro
   sbatch gromacs_checkpoint.sbatch
   squeue -u $USER
   tail -f gmx_checkpoint_JOBID.err
   ```

2. After about 90 seconds `mdrun` reaches its time limit and stops cleanly:

   ```
   starting mdrun 'Pure Water'
   5000000 steps,   5000.0 ps.

   Step 24390: Run time exceeded 0.025 hours, will terminate the run within 10 steps
   ```

3. Check that the checkpoint was written:

   ```bash
   cat gmx_checkpoint_JOBID.out
   ls -lh md.cpt
   ```

   ```
   Job ID : 15254
   Node   : holy7c26401.rc.fas.harvard.edu
   Start  : Wed Sep 16 22:46:58 EDT 2026
   Stopped/finished: Wed Sep 16 22:48:29 EDT 2026
   -rw-r--r--. 1 pkrastev rc_admin 340K Sep 16 22:48 md.cpt
   ```

### 2. Restart from the checkpoint — `gromacs_restart.sbatch` (batch)

The restart script runs the same `mdrun` command with `-cpi md.cpt` added. It exits
with an error if `md.tpr` or `md.cpt` is missing. Because it uses the same
`-deffnm md`, output is appended to the existing `md.log` and `md.edr`.

1. Submit and follow the job:

   ```bash
   sbatch gromacs_restart.sbatch
   tail -f gmx_restart_JOBID.err
   ```

2. The log shows that the run continues from the checkpoint instead of step 0:

   ```
   starting mdrun 'Pure Water'
   5000000 steps,   5000.0 ps (continuing from step 24400,     24.4 ps).

   Step 47910: Run time exceeded 0.025 hours, will terminate the run within 10 steps
   ```

   The first run stopped at step 24,390 and the checkpoint is at step 24,400, so the
   restart picks up there and runs to step 47,910 before its own 90-second limit.
   It writes a new `md.cpt`, so you can submit `gromacs_restart.sbatch` again to
   continue further.

### Cleanup

```bash
rm -f md.cpt md_prev.cpt md.edr md.log md.gro '#md.'*'#'
rm -f gmx_checkpoint_*.out gmx_checkpoint_*.err gmx_restart_*.out gmx_restart_*.err
```

---

## References

- GROMACS: <https://www.gromacs.org/>
- GROMACS documentation: <https://manual.gromacs.org/current/>
- `gmx mdrun` (`-cpt`, `-cpi`, `-maxh`): <https://manual.gromacs.org/current/onlinehelp/gmx-mdrun.html>
- `gmx grompp`: <https://manual.gromacs.org/current/onlinehelp/gmx-grompp.html>
- Managing long simulations: <https://manual.gromacs.org/current/user-guide/managing-simulations.html>
