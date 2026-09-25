# Exercise 6 — Checkpointing in PyTorch

## Introduction

Training a neural network is a long loop over epochs, so it needs the same
protection as any long computation: a checkpoint to restart from. This exercise
shows what a training checkpoint has to contain, in three steps of increasing
completeness:

```
train → save checkpoint → [interruption] → restart → load checkpoint → keep training
```

The vehicle is a small convolutional network (two convolution layers, dropout and
two linear layers) trained on the MNIST handwritten-digit data with the Adam
optimizer, on a GPU. The exercise covers three versions of the training script:
no checkpointing, a basic checkpoint, and a complete training checkpoint. Each is
run **interactively** first and then in **batch** mode on the `gpu_test` partition.

Checkpointing is done at the application level: after every completed epoch the
script saves a dictionary with `torch.save()` to a temporary file and then
atomically renames it over the previous checkpoint, so an interruption never
leaves a corrupt checkpoint. The versions differ in what the dictionary holds:

| Version | Checkpoint contents |
|---------|---------------------|
| Basic (`mnist_checkpoint.py`) | completed epoch, model parameters, optimizer state |
| Complete (`mnist_complete_checkpoint.py`) | the above plus learning-rate scheduler state, best validation loss, early-stopping counter, CPU and CUDA random-number-generator states, and the state of the data-shuffling generator |

Key ideas along the way:

- **Model and optimizer state** — the optimizer (Adam) keeps running statistics
  for every parameter, so restarting from the model weights alone is not the same
  as continuing the run.
- **Training-loop state** — the scheduler, best validation loss and
  early-stopping counter decide when the learning rate drops and when training
  stops. Without them a restarted run behaves differently.
- **Random-number state** — restoring the RNG states makes the data order and
  dropout continue as in the original run. On a GPU some kernels are
  non-deterministic, so a resumed run matches an uninterrupted one closely but not
  bit for bit.
- **Epoch granularity** — checkpoints are written at the end of an epoch, so an
  interrupted epoch is repeated after a restart.

---

## Content

| File | Description | Mode |
|------|-------------|------|
| `mnist_naive.py` | Baseline training, no checkpointing | Interactive |
| `mnist_checkpoint.py` | Basic checkpoint after every epoch; manual `--resume` | Interactive |
| `mnist_complete_checkpoint.py` | Complete training checkpoint with scheduler, early stopping and RNG state; `--resume`, `--device`, `--seed` | Interactive |
| `mnist_naive.sbatch` | Runs `mnist_naive.py` on a GPU | Batch |
| `mnist_basic_checkpoint.sbatch` | Runs `mnist_checkpoint.py` until its 1-minute time limit stops it | Batch |
| `mnist_basic_restart.sbatch` | Resumes `mnist_checkpoint.py` from its checkpoint | Batch |
| `mnist_complete_checkpoint.sbatch` | Runs `mnist_complete_checkpoint.py` until its 1-minute time limit stops it | Batch |
| `mnist_complete_restart.sbatch` | Resumes `mnist_complete_checkpoint.py` from its checkpoint | Batch |
| `mnist_naive_48614512.out` | Example output of `mnist_naive.sbatch` (10 epochs, completed) | Batch output |
| `mnist_basic_checkpoint_48615102.out` | Example output of `mnist_basic_checkpoint.sbatch` (stopped by the time limit after epoch 16) | Batch output |
| `mnist_basic_restart_48615162.out` | Example output of `mnist_basic_restart.sbatch` (resumed at epoch 17, completed) | Batch output |
| `mnist_complete_checkpoint_48615586.out` | Example output of `mnist_complete_checkpoint.sbatch` (stopped by the time limit after epoch 6) | Batch output |
| `mnist_complete_restart_48615590.out` | Example output of `mnist_complete_restart.sbatch` (resumed at epoch 7, early stopping at epoch 15) | Batch output |

---

## Workflow

### Setup

Steps 1–3 run on a GPU node, not a login node. All scripts accept
`--device auto|cpu|cuda`; the examples use `--device cuda`.

```bash
salloc --partition=gpu_test --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00
module load python
mamba activate /n/holylabs/rc_admin/Everyone/checkpoint-training/pt2.14.0_cuda13.2
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

```
2.14.0+cu132 True
```

The first run downloads MNIST (about 12 MB) into `./data`. In all interactive
examples `--sleep 2` pauses after each epoch, so there is time to interrupt the
run.

### 1. No checkpointing — `mnist_naive.py` (interactive)

Training progress lives only in memory, so an interruption loses everything.

1. Start the training:

   ```bash
   python mnist_naive.py --epochs 10 --sleep 2 --device cuda
   ```

   ```
   ============================================================
   PyTorch MNIST — baseline training
   ============================================================
   PyTorch version : 2.14.0+cu132
   Device          : cuda:0
   GPU             : NVIDIA A100 80GB PCIe
   Epochs          : 10
   ...
   Checkpointing   : disabled
   ============================================================

   Epoch 1/10 | train loss 0.1709 | train acc 94.70% | val loss 0.0422 | val acc 98.63% | time 5.32 s
   Epoch 2/10 | train loss 0.0616 | train acc 98.13% | val loss 0.0348 | val acc 98.92% | time 2.38 s
   Epoch 3/10 | train loss 0.0450 | train acc 98.63% | val loss 0.0351 | val acc 98.91% | time 2.64 s
   ...
   ```

2. Press `Ctrl-C` after a few epochs.
3. Rerun the same command. It starts again from epoch 1 — nothing was saved.

### 2. Basic checkpoint — `mnist_checkpoint.py` (interactive)

After every epoch the script saves the epoch number, the model parameters and the
optimizer state to `mnist_checkpoint.pt` (about 38 MB). With `--resume` it loads
them and continues with the next epoch.

1. Remove any old checkpoint:

   ```bash
   rm -f mnist_checkpoint.pt mnist_checkpoint.pt.tmp
   ```

2. Start the training:

   ```bash
   python mnist_checkpoint.py --epochs 10 --sleep 2 --device cuda
   ```

   ```
   ...
   Epoch 3/10 | train loss 0.0451 | train acc 98.61% | val loss 0.0293 | val acc 99.05% | time 2.48 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 3)
   Epoch 4/10 | train loss 0.0379 | train acc 98.82% | val loss 0.0290 | val acc 99.02% | time 2.50 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 4)
   ...
   ```

3. After a few checkpoints, press `Ctrl-C`.
4. Resume with the same parameters plus `--resume`:

   ```bash
   python mnist_checkpoint.py --epochs 10 --sleep 2 --device cuda --resume
   ```

   ```
   Loading checkpoint: mnist_checkpoint.pt
   Checkpoint restored from epoch 4.
   Training will resume at epoch 5.

   Epoch 5/10 | train loss 0.0317 | train acc 98.98% | val loss 0.0268 | val acc 99.17% | time 5.94 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 5)
   ...
   ```

   Training continues from epoch 5 instead of epoch 1. This checkpoint holds no
   random-number or scheduler state, so the data order after the restart is not
   the one the original run would have used. Step 3 adds that state.

### 3. Complete training checkpoint — `mnist_complete_checkpoint.py` (interactive)

This version adds a learning-rate schedule and early stopping.
`ReduceLROnPlateau` halves the learning rate when the validation loss has not
improved for more than 2 epochs, and training stops after 5 epochs in a row
without improvement. Its checkpoint adds the scheduler state, best validation
loss, early-stopping counter and the RNG states, so a resumed run continues where
these mechanisms stood.

1. Remove any old checkpoint:

   ```bash
   rm -f mnist_complete.pt mnist_complete.pt.tmp
   ```

2. Start the training:

   ```bash
   python mnist_complete_checkpoint.py --epochs 20 --sleep 2 --device cuda
   ```

   ```
   PyTorch MNIST — complete training checkpoint
   ============================================================
   Selected device       : cuda:0
   GPU                   : NVIDIA A100 80GB PCIe
   Epochs                : 20
   ...
   Early-stop patience   : 5
   Checkpoint            : mnist_complete.pt
   Resume                : False
   Seed                  : 12345
   ...
   Epoch 3/20 | train loss 0.0456 | train acc 98.57% | val loss 0.0268 | val acc 99.11% | lr 0.001 | improved yes | patience 0/5 | time 2.26 s
   Checkpoint saved: mnist_complete.pt (after epoch 3)
   Epoch 4/20 | train loss 0.0381 | train acc 98.81% | val loss 0.0310 | val acc 99.02% | lr 0.001 | improved no | patience 1/5 | time 2.42 s
   Checkpoint saved: mnist_complete.pt (after epoch 4)
   ...
   ```

3. After a few epochs, press `Ctrl-C`.
4. Resume with the same parameters plus `--resume`:

   ```bash
   python mnist_complete_checkpoint.py --epochs 20 --sleep 2 --device cuda --resume
   ```

   ```
   Loading checkpoint: mnist_complete.pt
   CUDA RNG state restored.
   Checkpoint restored from epoch 4.
   Training will resume at epoch 5.
   Checkpoint was saved from device type: cuda
   Current device: cuda:0
   Best validation loss: 0.026787
   Early-stopping counter: 1
   Current learning rate: 0.001

   Epoch 5/20 | train loss 0.0326 | train acc 98.93% | val loss 0.0301 | val acc 99.08% | lr 0.001 | improved no | patience 2/5 | time 5.26 s
   Checkpoint saved: mnist_complete.pt (after epoch 5)
   Learning rate reduced: 0.001 -> 0.0005
   ```

   The early-stopping counter (1) and best validation loss carry over, so the
   patience count continues (2/5), and the scheduler state is restored as well.
   If a checkpoint has already reached early stopping, `--resume` reports that
   there is nothing left to train.

The checkpoint is loaded onto the CPU first, so it can also be resumed on another
device, for example `--device cpu` for a run started on the GPU. That works but is
slow.

### 4. Batch mode on the `gpu_test` partition

The same three cases run in batch mode with these scripts. Each requests one GPU,
four CPUs and 16 GB on the `gpu_test` partition:

- `mnist_naive.sbatch` runs 10 epochs and completes.
- The `*_checkpoint.sbatch` scripts use `--sleep` and a 1-minute time limit, so the
  training is longer than the job: Slurm stops it (state `TIMEOUT`) and the last
  checkpoint remains. They remove any old checkpoint first.
- The `*_restart.sbatch` scripts run the same command with `--resume`. They stop
  with an error if the checkpoint is missing.

1. Run the training without checkpointing:

   ```bash
   sbatch mnist_naive.sbatch
   squeue -u $USER
   tail -f mnist_naive_JOBID.out
   ```

   From `mnist_naive_48614512.out`:

   ```
   ============================================================
   MNIST training on GPU - no checkpointing
   ============================================================
   Job ID : 48614512
   Node   : holygpu7c26105.rc.fas.harvard.edu
   GPU    : NVIDIA A100-SXM4-40GB
   Start  : Fri Sep 25 18:00:13 EDT 2026
   ============================================================
   ============================================================
   PyTorch MNIST — baseline training
   ============================================================
   PyTorch version : 2.14.0+cu132
   Device          : cuda:0
   GPU             : NVIDIA A100-SXM4-40GB MIG 3g.20gb
   Epochs          : 10
   ...
   Epoch 1/10 | train loss 0.1753 | train acc 94.57% | val loss 0.0419 | val acc 98.58% | time 6.89 s
   ...
   Epoch 10/10 | train loss 0.0163 | train acc 99.45% | val loss 0.0299 | val acc 99.23% | time 2.04 s

   ============================================================
   Training complete
   ============================================================
   Total time : 25.16 s
   ============================================================
   End: Fri Sep 25 18:00:57 EDT 2026
   ```

   The GPU on `gpu_test` is a MIG slice (`3g.20gb`) of an A100, which is ample
   for this small model.

2. Basic checkpoint: submit the checkpoint job and the restart job. The
   dependency starts the restart job once the first has ended (replace `JOBID`
   with the job ID printed by `sbatch`):

   ```bash
   JOBID=$(sbatch --parsable mnist_basic_checkpoint.sbatch)
   sbatch --dependency=afterany:$JOBID mnist_basic_restart.sbatch
   ```

   The first job is cut off at its time limit (Slurm records the cancellation
   `DUE TO TIME LIMIT` in `mnist_basic_checkpoint_JOBID.err`). Its output ends
   with the last checkpoint, with no `End:` line
   (`mnist_basic_checkpoint_48615102.out`):

   ```
   Epoch 15/20 | train loss 0.0120 | train acc 99.58% | val loss 0.0299 | val acc 99.25% | time 2.03 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 15)
   ...
   Epoch 16/20 | train loss 0.0123 | train acc 99.56% | val loss 0.0304 | val acc 99.25% | time 2.03 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 16)
   ```

   The second job resumes from the last completed epoch (here epoch 16) and
   finishes the 20 epochs (`mnist_basic_restart_48615162.out`):

   ```
   Loading checkpoint: mnist_checkpoint.pt
   Checkpoint restored from epoch 16.
   Training will resume at epoch 17.
   ...
   Epoch 17/20 | train loss 0.0109 | train acc 99.63% | val loss 0.0295 | val acc 99.32% | time 4.20 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 17)
   ...
   Epoch 20/20 | train loss 0.0103 | train acc 99.67% | val loss 0.0294 | val acc 99.30% | time 2.11 s
   Checkpoint saved: mnist_checkpoint.pt (after epoch 20)

   ============================================================
   Training complete
   ============================================================
   Total time : 20.13 s
   Checkpoint : mnist_checkpoint.pt
   ============================================================
   End: Fri Sep 25 18:05:41 EDT 2026
   ```

3. Complete checkpoint: the same, with the complete-checkpoint scripts:

   ```bash
   JOBID=$(sbatch --parsable mnist_complete_checkpoint.sbatch)
   sbatch --dependency=afterany:$JOBID mnist_complete_restart.sbatch
   ```

   `--sleep 8` slows this job to about 10 seconds per epoch, so the 1-minute limit
   stops it after about 6 epochs, before early stopping can trigger
   (`mnist_complete_checkpoint_48615586.out`):

   ```
   Epoch 5/20 | train loss 0.0321 | train acc 98.97% | val loss 0.0283 | val acc 99.02% | lr 0.001 | improved yes | patience 0/5 | time 1.68 s
   Checkpoint saved: mnist_complete.pt (after epoch 5)
   ...
   Epoch 6/20 | train loss 0.0254 | train acc 99.18% | val loss 0.0271 | val acc 99.08% | lr 0.001 | improved yes | patience 0/5 | time 1.72 s
   Checkpoint saved: mnist_complete.pt (after epoch 6)
   ```

   The restart job restores the full state and continues
   (`mnist_complete_restart_48615590.out`):

   ```
   Loading checkpoint: mnist_complete.pt
   CUDA RNG state restored.
   Checkpoint restored from epoch 6.
   Training will resume at epoch 7.
   Checkpoint was saved from device type: cuda
   Current device: cuda:0
   Best validation loss: 0.027076
   Early-stopping counter: 0
   Current learning rate: 0.001
   ...
   Epoch 7/20 | train loss 0.0241 | train acc 99.19% | val loss 0.0299 | val acc 99.08% | lr 0.001 | improved no | patience 1/5 | time 3.96 s
   ...
   Learning rate reduced: 0.001 -> 0.0005

   Epoch 13/20 | train loss 0.0140 | train acc 99.55% | val loss 0.0278 | val acc 99.23% | lr 0.0005 | improved no | patience 3/5 | time 1.79 s
   ...
   Epoch 15/20 | train loss 0.0053 | train acc 99.83% | val loss 0.0269 | val acc 99.40% | lr 0.0005 | improved no | patience 5/5 | time 1.78 s
   Checkpoint saved: mnist_complete.pt (after epoch 15)

   Early stopping triggered.
   No validation-loss improvement for 5 consecutive epochs.

   ============================================================
   Training finished
   ============================================================
   Best validation loss : 0.024100
   Final learning rate  : 0.0005
   Total time           : 86.39 s
   Checkpoint           : mnist_complete.pt
   ============================================================
   End: Fri Sep 25 18:09:55 EDT 2026
   ```

   The best validation loss (0.027076) and early-stopping counter carried over
   from the first job. The learning rate was halved after epoch 13, and early
   stopping triggered after epoch 15, five epochs after the best validation loss
   (0.024100, epoch 10), so the run ended before the 20 requested epochs.

4. Check the outcome of the jobs:

   ```bash
   sacct -j JOBID -X -o JobID,JobName%22,State,Elapsed
   ```

   For the five example jobs:

   ```
   JobID                       JobName      State    Elapsed
   ------------ ---------------------- ---------- ----------
   48614512                mnist-naive  COMPLETED   00:00:46
   48615102           mnist-basic-ckpt    TIMEOUT   00:01:21
   48615162        mnist-basic-restart  COMPLETED   00:00:29
   48615586        mnist-complete-ckpt    TIMEOUT   00:01:09
   48615590     mnist-complete-restart  COMPLETED   00:01:36
   ```

   The checkpoint jobs show `TIMEOUT` and the restart jobs `COMPLETED`. The exact
   epoch at which a job is cut off varies from run to run.

### Cleanup

Leave the allocation, then remove the generated files. Note that
`mnist_*_*.out` also matches the example output files listed under Content:

```bash
exit
rm -f mnist_checkpoint.pt mnist_checkpoint.pt.tmp mnist_complete.pt mnist_complete.pt.tmp
rm -f mnist_*_*.out mnist_*_*.err
rm -rf data
```

---

## References

- PyTorch, saving and loading models: <https://pytorch.org/tutorials/beginner/saving_loading_models.html>
- PyTorch, saving and loading a general checkpoint: <https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html>
- PyTorch, reproducibility: <https://docs.pytorch.org/docs/stable/notes/randomness.html>
