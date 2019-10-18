### Running Pytorch examples on FASRC Cluster

##### 1. Get an interactive node
        srun --pty -p gpu_test -t 0-03:00 --mem 4000 --gres=gpu:1 /bin/bash
##### 2. Module load Anaconda and CUDA
module load Anaconda3/5.0.1-fasrc02  
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01

##### 3.  Conda install
        conda info --envs
        conda create -n pytorch_3  pytorch torchvision cudatoolkit=10.0 matplotlib  
##### 4. Test module
        source activate pytorch_3

##### 5. Download examples
	    git clone https://github.com/pytorch/examples

##### 6. Use as a batch job
		vi pytorch.slurm 
Update the batch script bases on your test. 
```
#!/bin/bash
##SBATCH -n 1               # Number of cores
##SBATCH -N 1               # Ensure that all cores are on one machine
#SBATCH -t 0-00:30          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu              # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=4G            # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o pytorch_%j.out   # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e pytorch_%j.err   # File to which STDERR will be written, %j inserts jobid

module load Anaconda3/5.0.1-fasrc02
module load cuda/10.0.130-fasrc01 cudnn/7.4.1.5_cuda10.0-fasrc01

source activate pytorch_3

python examples/mnist/main.py
```
