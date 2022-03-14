## Dynamite

[Dynamite](https://dynamite.readthedocs.io/en/latest/index.html) provides a simple interface to fast evolution of quantum dynamics and eigensolving. Behind the scenes, dynamite uses the PETSc/SLEPc implementations of Krylov subspace exponentiation and eigensolving.

### Setting up Dynamite on the FASRC Cannon Cluster

Below steps outline how to setup Dynamite on the Cannon cluster in your environment.

#### (1) Load the required software modules

```bash
module load python/3.8.5-fasrc01
module load gcc/10.2.0-fasrc01   
module load openmpi/4.1.1-fasrc01
```
#### (2) Download dynamite and pull the development branch

```bash
mkdir ~/sw
cd sw/
git clone https://github.com/GregDMeyer/dynamite.git
cd dynamite/
git pull origin dev
```

#### (3) Download and install PETSc

```bash
cd ~/sw
git clone https://gitlab.com/petsc/petsc.git petsc
cd petsc
git checkout tags/v3.15.0
```

**Note:** You will need to modify the script <code>~/sw/dynamite/petsc_config/complex-opt.py</code> to add the configure option <code>--with-64-bit-indices=1</code>. This is required, if PETSc is intended to be used for very large matrices. Then, you need to configure PETSc with the below commands:

```bash
cd ~/sw/petsc
python ../dynamite/petsc_config/complex-opt.py
```

If all goes well, configure will tell you to run a <code>make</code> command. Copy the command and run it. It should look like: 

```bash
make PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt all
```

#### (4) Download and install SLEPc

```bash
cd ~/sw
export PETSC_DIR=<your_petsc_directory> PETSC_ARCH=complex-opt
git clone https://gitlab.com/slepc/slepc.git slepc
cd slepc
git checkout tags/v3.15.0
./configure
```

If it configures correctly, it will output a <code>make</code> command to run. Copy and paste that, and run it. It should look like:

```bash
make SLEPC_DIR=$PWD PETSC_DIR=<petsc_dir> PETSC_ARCH=complex-opt
```

#### (5) Building Dynamite

Dynamite is best installed in a <code>conda</code> environment. After you create your environment, you will need to activate it, e.g.:

```bash
conda create -n dynamite python=3.8 pip wheel cython
source activate dynamite
```

You will need to make sure that these environmental variables are all set correctly:

```bash
export PETSC_DIR=<petsc_dir> 
export PETSC_ARCH=complex-opt
export SLEPC_DIR=<your_slepc_installation_directory>
```

Next, install the required Python packages from <copy>requirements.txt</copy>. **Please, notice that here we use the instance from the release dynamite version.**

```bash
cd ~/sw/dynamite
# Get the requirements.txt from the release version
wget https://raw.githubusercontent.com/GregDMeyer/dynamite/master/requirements.txt
pip install -r requirements.txt
```

Then build dynamite:

```bash
cd ~/sw/dynamite
pip install -e ./
```

### Example Python Script:

```python
import argparse 
import numpy as np
from time import time
from scipy.special import comb
from mpi4py import MPI

from dynamite import config
from dynamite.subspaces import SpinConserve
#from dynamite.subspaces import *
from dynamite.operators import sigmax, sigmay, sigmaz, op_sum, index_product
from dynamite.msc_tools import *
from dynamite.tools import *


# Run the Heisenberg model on the Kagome lattice
# Requires the number of spins and the number of up spins

# Saves the eigenvector as a binary file
def main():
    parser = argparse.ArgumentParser(description="Run Heisenberg on the kagome lattice.")
    # Set the default for the dataset argument
    parser.add_argument('--N', dest='N', default="12", type=str)
    parser.add_argument('--num_up_spins', dest='U', default=6, type=int)
    parser.add_argument('--shell', default=False, action='store_true')
    parser.add_argument('--spinflip', default='None', type=str)
    parser.add_argument('--monitor', default=False, action='store_true')
    args = parser.parse_args()

    N = int(''.join(i for i in args.N if i.isdigit()))
    num_up = args.U

    # Load the edges and the ground state energy
    dir_path = "/n/home06/pkrastev/rc_team/tests/dynamite_test/kagome/"
    edges_file_path = dir_path + "data/edges%s.txt"%(args.N)
    #gs_file_path = dir_path + "data/gs%s.txt"%(args.N)

    edges = np.loadtxt(edges_file_path, dtype=int)
    #gs = np.loadtxt(gs_file_path, dtype=float)

    # Whether to print iteration info using the PETSc monitor
    if args.monitor:
        config.initialize(['-eps_monitor'])#, '-eps_view', '-log_view'])
    
    # Parse spinflip option
    spinflip = None
    if args.spinflip == "+" or args.spinflip == "-":
        spinflip = args.spinflip
    elif args.spinflip != "None":
        raise ValueError("spinflip is either + or -")
    # Create hamiltonian
    config.L = int(N)
    h = 1/4*op_sum(sigmax(i)*sigmax(j) + \
               sigmay(i)*sigmay(j) + \
               sigmaz(i)*sigmaz(j) for i, j in edges)

    # Whether to use shell mode
    h.shell = args.shell
    tic = time()
    subspace = SpinConserve(config.L, num_up, spinflip=spinflip)
    toc = time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        print("Subspace (s): ", toc - tic, flush=True)

    h.subspace = subspace
    
    #before = get_cur_memory_usage()
    tic = time()
    #energies, eigvecs = h.eigsolve(getvecs=True)
    energies = h.eigsolve()
    toc = time()
    #after = get_cur_memory_usage()
    '''
    from petsc4py import PETSc
    
    # Save the eigenvector
    eigvec_file_path = dir_path + "heis_%s_vec.dat"%(args.N)
    w_viewer = PETSc.Viewer().createBinary(eigvec_file_path, 'w',
               comm=PETSc.COMM_WORLD)
    eigvecs[0].vec.view(w_viewer)

    # Check 
    r_viewer = PETSc.Viewer().createBinary(eigvec_file_path, 'r',
               comm=PETSc.COMM_WORLD)
    u = PETSc.Vec().load(r_viewer)
    assert eigvecs[0].vec.equal(u)
    '''
    if rank == 0:
        print("Eigsolve (s): ", toc - tic)
        print("lowest energy: %f" %(energies[0]))
        #print("expected energy: %f" %(gs))
        #print("relative error: %.2e" %(np.abs((energies[0] - gs)/gs)))
        #print('matrix memory usage: %f Mb' % ((after-before)/1E6))
  
if __name__ == "__main__":
    main()
```

### Example Batch-Job Submission Script:

```bash
#!/bin/bash -l
#SBATCH -J N36_U18_n512_shell
#SBATCH -t 2-00:00
#SBATCH -o n512_shell.out
#SBATCH -e n512_shell.err
#SBATCH -p shared
#SBATCH -n 512
#SBATCH -N 64
#SBATCH --ntasks-per-node=8
#SBATCH --mem-per-cpu=8G
#SBATCH -x holy2a19307

# Set up environment
module load python/3.8.5-fasrc01 
module load gcc/10.2.0-fasrc01
module load openmpi/4.1.1-fasrc01
source activate dynamite
export PETSC_DIR=<PETSC_LOCATION>/petsc 
export PETSC_ARCH=complex-opt
export SLEPC_DIR=<SLEPC_LOCATION>/slepc

# Run program
time srun -n 512 --mpi=pmix  python run_heis_kagome_spinconserve.py --N 36 --num_up_spins 18 --shell --monitor
```

### Example Output:

```
Subspace (s):  0.0006227493286132812
  1 EPS nconv=0 first unconverged value (error) -15.032+0i (6.15452408e-02)
  2 EPS nconv=0 first unconverged value (error) -15.5044+0i (3.26595719e-02)
  3 EPS nconv=0 first unconverged value (error) -15.6523+0i (1.95111880e-02)
  4 EPS nconv=0 first unconverged value (error) -15.7129+0i (1.32490815e-02)
  5 EPS nconv=0 first unconverged value (error) -15.7409+0i (9.81638414e-03)
  6 EPS nconv=0 first unconverged value (error) -15.7595+0i (9.74307026e-03)
  7 EPS nconv=0 first unconverged value (error) -15.7772+0i (8.85957623e-03)
  8 EPS nconv=0 first unconverged value (error) -15.7913+0i (7.75999297e-03)
  9 EPS nconv=0 first unconverged value (error) -15.8019+0i (6.26800697e-03)
 10 EPS nconv=0 first unconverged value (error) -15.8085+0i (4.76529153e-03)
 11 EPS nconv=0 first unconverged value (error) -15.8117+0i (3.22744776e-03)
 12 EPS nconv=0 first unconverged value (error) -15.813+0i (2.26363678e-03)
 13 EPS nconv=0 first unconverged value (error) -15.8136+0i (1.51102158e-03)
 14 EPS nconv=0 first unconverged value (error) -15.8138+0i (1.09526205e-03)
 15 EPS nconv=0 first unconverged value (error) -15.814+0i (8.28988692e-04)
 16 EPS nconv=0 first unconverged value (error) -15.8141+0i (7.24207126e-04)
 17 EPS nconv=0 first unconverged value (error) -15.8141+0i (6.42843428e-04)
 18 EPS nconv=0 first unconverged value (error) -15.8142+0i (6.10340814e-04)
 19 EPS nconv=0 first unconverged value (error) -15.8142+0i (5.51641730e-04)
 20 EPS nconv=0 first unconverged value (error) -15.8143+0i (4.98788862e-04)
 21 EPS nconv=0 first unconverged value (error) -15.8143+0i (4.06420038e-04)
 22 EPS nconv=0 first unconverged value (error) -15.8143+0i (3.30687878e-04)
 23 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.52598133e-04)
 24 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.99440039e-04)
 25 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.49571026e-04)
 26 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.16663354e-04)
 27 EPS nconv=0 first unconverged value (error) -15.8143+0i (8.77298592e-05)
 28 EPS nconv=0 first unconverged value (error) -15.8143+0i (6.99765242e-05)
 29 EPS nconv=0 first unconverged value (error) -15.8143+0i (5.43091765e-05)
 30 EPS nconv=0 first unconverged value (error) -15.8143+0i (4.43432245e-05)
 31 EPS nconv=0 first unconverged value (error) -15.8143+0i (3.45173686e-05)
 32 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.76235524e-05)
 33 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.07241727e-05)
 34 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.59162617e-05)
 35 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.15626945e-05)
 36 EPS nconv=0 first unconverged value (error) -15.8143+0i (8.74722215e-06)
 37 EPS nconv=0 first unconverged value (error) -15.8143+0i (6.37374691e-06)
 38 EPS nconv=0 first unconverged value (error) -15.8143+0i (4.90428302e-06)
 39 EPS nconv=0 first unconverged value (error) -15.8143+0i (3.66103867e-06)
 40 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.88797896e-06)
 41 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.20003865e-06)
 42 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.75736741e-06)
 43 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.34336587e-06)
 44 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.06738181e-06)
 45 EPS nconv=0 first unconverged value (error) -15.8143+0i (8.05849515e-07)
 46 EPS nconv=0 first unconverged value (error) -15.8143+0i (6.29479843e-07)
 47 EPS nconv=0 first unconverged value (error) -15.8143+0i (4.66446864e-07)
 48 EPS nconv=0 first unconverged value (error) -15.8143+0i (3.57980820e-07)
 49 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.61422701e-07)
 50 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.98564902e-07)
 51 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.44179370e-07)
 52 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.09379673e-07)
 53 EPS nconv=0 first unconverged value (error) -15.8143+0i (7.96447419e-08)
 54 EPS nconv=0 first unconverged value (error) -15.8143+0i (6.07999903e-08)
 55 EPS nconv=0 first unconverged value (error) -15.8143+0i (4.46717754e-08)
 56 EPS nconv=0 first unconverged value (error) -15.8143+0i (3.44850292e-08)
 57 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.56610271e-08)
 58 EPS nconv=0 first unconverged value (error) -15.8143+0i (2.00827677e-08)
 59 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.51564930e-08)
 60 EPS nconv=0 first unconverged value (error) -15.8143+0i (1.20295852e-08)
 61 EPS nconv=1 first unconverged value (error) -15.8025+0i (7.10033299e-07)
Eigsolve (s):  91833.94949650764
lowest energy: -15.814334
```

### References:

* [Official Dynamite Documentation](https://dynamite.readthedocs.io)
* [Examples](https://github.com/GregDMeyer/dynamite/tree/master/examples)



