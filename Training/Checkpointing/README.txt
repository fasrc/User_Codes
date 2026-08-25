README

 - Run naive ( no checkpointing )

module load python

python pi_naive.py --darts 100000000 --seed 42 --report-every 1000000

python pi_naive.py \
    -n 100000000 \
    --report-every 1000000 \
    --seed 42 \
    --sleep 1

- Run checkpointed ( no SIGNAL yet !!! )

	+++ Run +++

python pi_checkpoint.py \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --sleep 1

	+++ Restart +++

python pi_checkpoint.py \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --sleep 1 \
    --resume

                CHECKPOINT
                    |
       +------------+-------------+
       |            |             |
   progress      numerical       RNG
     state         state         state
       |            |             |
 completed      inside_circle   getstate()
   darts

- Run SLURM SIGNAL

python pi_signal.py \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --sleep 1

From another terminal:

kill -USR1 123456

python pi_signal.py \
    --darts 100000000 \
    --seed 42 \
    --report-every 1000000 \
    --checkpoint-every 5000000 \
    --sleep 1 \
    --resume


