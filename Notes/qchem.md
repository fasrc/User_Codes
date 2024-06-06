# Q-Chem

[Q-Chem](https://www.q-chem.com/) is an ab initio quantum chemistry software package for fast and accurate simulations of molecular systems, including electronic and molecular structure, reactivities, properties, and spectra.

## Q-Chem on FAS Cannon Cluster

Q-Chem is available on the Cannon cluster as a software module - `QChem/6.1-fasrc01`. Due to license restrictions, the packages is currently available only to members of the Department of Chemistry and Chemical Biology at Harvard University.

### Using Q-Chem

**Example batch-job submission script**

```bash
#!/bin/bash
#SBATCH -J qchem_test
#sbatch -N 1
#SBATCH -c 8
#SBATCH -e qchem.err 
#SBATCH -o qchem.out
#SBATCH -p test
#SBATCH --mem=32G

# Load the required software modules
module load QChem/6.1-fasrc01

# Run the program
qchem -np 8 qchem_test.in qchem_test.out
```

If the above submission script is named `run.sbatch`, for instance, the job is sent to the queue with:

```bash
sbatch run.sbatch
```

**Example input file `qchem_test.in`**

The above submission script assumes that there is a file named `qchem_test.in` in the work directory (from where the job is submitted) with the below contents:

```
$molecule
  0 1
  O
  H1 O oh
  H2 O oh H1 hoh

  oh  = 1.2
  hoh = 120.0
$end

$rem
  JOBTYPE     sp       Single Point energy
  EXCHANGE    hf       Exact HF exchange
  CORRELATION none     No correlation
  BASIS       sto-3g   Basis set
$end

$comment
HF/STO-3G water single point calculation
$end
```

### References:
* [Q-Chem official website](https://www.q-chem.com/)
* [Q-Chem training materials](https://www.q-chem.com/learn/)