### Purpose:

Example of using HDF5 libraries on the cluster. The specific example creates a random vector and writes it to a HDF5 (.h5) file.

### Contents:

1. `hdf5_test.f90`: Fortran 90 source file

2. `Makefile`: Makefile to compile the source code 

3. `run_hdf5_test.sh`: Batch-job submission script to send the job to the queue

### Example Usage:

    module load gcc/14.2.0-fasrc01  openmpi/5.0.5-fasrc01 hdf5/1.14.4-fasrc01
	make
	sbatch hdf5_test.sbatch

### Example Output:

```
[jharvard@boslogin07 hdf5]$ h5dump output.h5
HDF5 "output.h5" {
GROUP "/" {
   DATASET "darr" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( 100 ) / ( 100 ) }
      DATA {
      (0): 0.19997, 0.141935, 0.9609, 0.557535, 0.638827, 0.462287, 0.498032,
      (7): 0.867238, 0.762328, 0.23642, 0.0605957, 0.150953, 0.948071,
      (13): 0.844535, 0.181702, 0.280175, 0.229308, 0.552097, 0.73992,
      (19): 0.331059, 0.388762, 0.41635, 0.325443, 0.412334, 0.631472,
      (25): 0.890654, 0.765588, 0.144551, 0.148686, 0.0065211, 0.367894,
      (31): 0.152968, 0.220681, 0.629017, 0.459204, 0.697648, 0.389048,
      (37): 0.919453, 0.199217, 0.58728, 0.363542, 0.754135, 0.194908,
      (43): 0.815831, 0.938569, 0.42343, 0.296155, 0.811725, 0.364881,
      (49): 0.893735, 0.00850734, 0.887926, 0.194576, 0.588769, 0.0942312,
      (55): 0.0470023, 0.921254, 0.331883, 0.0983307, 0.941179, 0.0732383,
      (61): 0.578579, 0.668021, 0.175048, 0.872878, 0.306461, 0.956044,
      (67): 0.13224, 0.905966, 0.758271, 0.98402, 0.417583, 0.187216,
      (73): 0.846184, 0.322552, 0.500836, 0.221774, 0.736674, 0.318103,
      (79): 0.584469, 0.9694, 0.433704, 0.0462207, 0.207506, 0.933283,
      (85): 0.789315, 0.484947, 0.045633, 0.756139, 0.152743, 0.741604,
      (91): 0.256809, 0.0134879, 0.440945, 0.603259, 0.94596, 0.566918,
      (97): 0.348724, 0.493279, 0.437733
      }
   }
}
}
```

### Troubleshooting

#### Querying hdf5 files on netscratch

If you are on a login node and would like to query hdf5 files located on [`/n/netscratch`](https://docs.rc.fas.harvard.edu/kb/data-storage-workflow-rdm/#Scratch) with, for example, `h5ls` and `h5dump`, you have to set:

```
export HDF5_USE_FILE_LOCKING=FALSE
```

Without this settings, the commands `h5ls` and `h5dump` will hang.
