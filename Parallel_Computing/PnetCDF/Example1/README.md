### Purpose:

This program tests PnetCDF parallel I/O functions from Fortran. The specific example shows how to use varm API to write six 3D integer array variables into a file. Each variable in the file is a dimensional transposed array from the one stored in memory. In memory, a 3D array is partitioned among all processes in a block-block-block fashion and in XYZ (i.e. Fortran) order. Note the variable and dimension naming below is in Fortran order.

### Contents:

* <code>transpose.f90</code>: Fortran source code
* <code>utils.F90</code>: Fortran source code
* <code>Makefile</code>: Makefile to compile this example
* <code>run.sbatch</code>: Batch-job submission script

### Example Batch-Job Submission Script:

```bash
#!/bin/bash
#SBATCH -J parallel_netcdf
#SBATCH -o parallel_netcdf.out
#SBATCH -e parallel_netcdf.err
#SBATCH -p general
#SBATCH -t 0-00:30
#SBATCH -n 8
#SBATCH --mem-per-cpu=4000

# Load required modules
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load parallel-netcdf/1.8.1-fasrc01

# Run program
srun -n 8 --mpi=pmi2 ./transpose.x
```

### Makefile:

```bash
#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpiifort
F90CFLAGS   = -c -O2
LIBS        = -lpnetcdf
PRO         = transpose

OBJECTS =  utils.o     \
           transpose.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

%.o : %.F90
	$(F90COMPILER) $(F90CFLAGS) $(<F)


clean : 
	rm -rf *.o *.x
```

### Example Usage:

```bash
source new-modules.sh
module load intel/17.0.2-fasrc01
module load impi/2017.2.174-fasrc01
module load parallel-netcdf/1.8.1-fasrc01
make
sbatch run.sbatch
```

### Example Output:

```
ncdump testfile.nc 
netcdf testfile {
dimensions:
	X = 8 ;
	Y = 6 ;
	Z = 4 ;
variables:
	int XYZ_var(Z, Y, X) ;
	int XZY_var(Y, Z, X) ;
	int YXZ_var(Z, X, Y) ;
	int YZX_var(X, Z, Y) ;
	int ZXY_var(Y, X, Z) ;
	int ZYX_var(X, Y, Z) ;
data:

 XYZ_var =
  0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23,
  24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35, 36, 37, 38, 39,
  40, 41, 42, 43, 44, 45, 46, 47,
  48, 49, 50, 51, 52, 53, 54, 55,
  56, 57, 58, 59, 60, 61, 62, 63,
  64, 65, 66, 67, 68, 69, 70, 71,
  72, 73, 74, 75, 76, 77, 78, 79,
  80, 81, 82, 83, 84, 85, 86, 87,
  88, 89, 90, 91, 92, 93, 94, 95,
  96, 97, 98, 99, 100, 101, 102, 103,
  104, 105, 106, 107, 108, 109, 110, 111,
  112, 113, 114, 115, 116, 117, 118, 119,
  120, 121, 122, 123, 124, 125, 126, 127,
  128, 129, 130, 131, 132, 133, 134, 135,
  136, 137, 138, 139, 140, 141, 142, 143,
  144, 145, 146, 147, 148, 149, 150, 151,
  152, 153, 154, 155, 156, 157, 158, 159,
  160, 161, 162, 163, 164, 165, 166, 167,
  168, 169, 170, 171, 172, 173, 174, 175,
  176, 177, 178, 179, 180, 181, 182, 183,
  184, 185, 186, 187, 188, 189, 190, 191 ;

 XZY_var =
  0, 1, 2, 3, 4, 5, 6, 7,
  48, 49, 50, 51, 52, 53, 54, 55,
  96, 97, 98, 99, 100, 101, 102, 103,
  144, 145, 146, 147, 148, 149, 150, 151,
  8, 9, 10, 11, 12, 13, 14, 15,
  56, 57, 58, 59, 60, 61, 62, 63,
  104, 105, 106, 107, 108, 109, 110, 111,
  152, 153, 154, 155, 156, 157, 158, 159,
  16, 17, 18, 19, 20, 21, 22, 23,
  64, 65, 66, 67, 68, 69, 70, 71,
  112, 113, 114, 115, 116, 117, 118, 119,
  160, 161, 162, 163, 164, 165, 166, 167,
  24, 25, 26, 27, 28, 29, 30, 31,
  72, 73, 74, 75, 76, 77, 78, 79,
  120, 121, 122, 123, 124, 125, 126, 127,
  168, 169, 170, 171, 172, 173, 174, 175,
  32, 33, 34, 35, 36, 37, 38, 39,
  80, 81, 82, 83, 84, 85, 86, 87,
  128, 129, 130, 131, 132, 133, 134, 135,
  176, 177, 178, 179, 180, 181, 182, 183,
  40, 41, 42, 43, 44, 45, 46, 47,
  88, 89, 90, 91, 92, 93, 94, 95,
  136, 137, 138, 139, 140, 141, 142, 143,
  184, 185, 186, 187, 188, 189, 190, 191 ;

 YXZ_var =
  0, 8, 16, 24, 32, 40,
  1, 9, 17, 25, 33, 41,
  2, 10, 18, 26, 34, 42,
  3, 11, 19, 27, 35, 43,
  4, 12, 20, 28, 36, 44,
  5, 13, 21, 29, 37, 45,
  6, 14, 22, 30, 38, 46,
  7, 15, 23, 31, 39, 47,
  48, 56, 64, 72, 80, 88,
  49, 57, 65, 73, 81, 89,
  50, 58, 66, 74, 82, 90,
  51, 59, 67, 75, 83, 91,
  52, 60, 68, 76, 84, 92,
  53, 61, 69, 77, 85, 93,
  54, 62, 70, 78, 86, 94,
  55, 63, 71, 79, 87, 95,
  96, 104, 112, 120, 128, 136,
  97, 105, 113, 121, 129, 137,
  98, 106, 114, 122, 130, 138,
  99, 107, 115, 123, 131, 139,
  100, 108, 116, 124, 132, 140,
  101, 109, 117, 125, 133, 141,
  102, 110, 118, 126, 134, 142,
  103, 111, 119, 127, 135, 143,
  144, 152, 160, 168, 176, 184,
  145, 153, 161, 169, 177, 185,
  146, 154, 162, 170, 178, 186,
  147, 155, 163, 171, 179, 187,
  148, 156, 164, 172, 180, 188,
  149, 157, 165, 173, 181, 189,
  150, 158, 166, 174, 182, 190,
  151, 159, 167, 175, 183, 191 ;

 YZX_var =
  0, 8, 16, 24, 32, 40,
  48, 56, 64, 72, 80, 88,
  96, 104, 112, 120, 128, 136,
  144, 152, 160, 168, 176, 184,
  1, 9, 17, 25, 33, 41,
  49, 57, 65, 73, 81, 89,
  97, 105, 113, 121, 129, 137,
  145, 153, 161, 169, 177, 185,
  2, 10, 18, 26, 34, 42,
  50, 58, 66, 74, 82, 90,
  98, 106, 114, 122, 130, 138,
  146, 154, 162, 170, 178, 186,
  3, 11, 19, 27, 35, 43,
  51, 59, 67, 75, 83, 91,
  99, 107, 115, 123, 131, 139,
  147, 155, 163, 171, 179, 187,
  4, 12, 20, 28, 36, 44,
  52, 60, 68, 76, 84, 92,
  100, 108, 116, 124, 132, 140,
  148, 156, 164, 172, 180, 188,
  5, 13, 21, 29, 37, 45,
  53, 61, 69, 77, 85, 93,
  101, 109, 117, 125, 133, 141,
  149, 157, 165, 173, 181, 189,
  6, 14, 22, 30, 38, 46,
  54, 62, 70, 78, 86, 94,
  102, 110, 118, 126, 134, 142,
  150, 158, 166, 174, 182, 190,
  7, 15, 23, 31, 39, 47,
  55, 63, 71, 79, 87, 95,
  103, 111, 119, 127, 135, 143,
  151, 159, 167, 175, 183, 191 ;

 ZXY_var =
  0, 48, 96, 144,
  1, 49, 97, 145,
  2, 50, 98, 146,
  3, 51, 99, 147,
  4, 52, 100, 148,
  5, 53, 101, 149,
  6, 54, 102, 150,
  7, 55, 103, 151,
  8, 56, 104, 152,
  9, 57, 105, 153,
  10, 58, 106, 154,
  11, 59, 107, 155,
  12, 60, 108, 156,
  13, 61, 109, 157,
  14, 62, 110, 158,
  15, 63, 111, 159,
  16, 64, 112, 160,
  17, 65, 113, 161,
  18, 66, 114, 162,
  19, 67, 115, 163,
  20, 68, 116, 164,
  21, 69, 117, 165,
  22, 70, 118, 166,
  23, 71, 119, 167,
  24, 72, 120, 168,
  25, 73, 121, 169,
  26, 74, 122, 170,
  27, 75, 123, 171,
  28, 76, 124, 172,
  29, 77, 125, 173,
  30, 78, 126, 174,
  31, 79, 127, 175,
  32, 80, 128, 176,
  33, 81, 129, 177,
  34, 82, 130, 178,
  35, 83, 131, 179,
  36, 84, 132, 180,
  37, 85, 133, 181,
  38, 86, 134, 182,
  39, 87, 135, 183,
  40, 88, 136, 184,
  41, 89, 137, 185,
  42, 90, 138, 186,
  43, 91, 139, 187,
  44, 92, 140, 188,
  45, 93, 141, 189,
  46, 94, 142, 190,
  47, 95, 143, 191 ;

 ZYX_var =
  0, 48, 96, 144,
  8, 56, 104, 152,
  16, 64, 112, 160,
  24, 72, 120, 168,
  32, 80, 128, 176,
  40, 88, 136, 184,
  1, 49, 97, 145,
  9, 57, 105, 153,
  17, 65, 113, 161,
  25, 73, 121, 169,
  33, 81, 129, 177,
  41, 89, 137, 185,
  2, 50, 98, 146,
  10, 58, 106, 154,
  18, 66, 114, 162,
  26, 74, 122, 170,
  34, 82, 130, 178,
  42, 90, 138, 186,
  3, 51, 99, 147,
  11, 59, 107, 155,
  19, 67, 115, 163,
  27, 75, 123, 171,
  35, 83, 131, 179,
  43, 91, 139, 187,
  4, 52, 100, 148,
  12, 60, 108, 156,
  20, 68, 116, 164,
  28, 76, 124, 172,
  36, 84, 132, 180,
  44, 92, 140, 188,
  5, 53, 101, 149,
  13, 61, 109, 157,
  21, 69, 117, 165,
  29, 77, 125, 173,
  37, 85, 133, 181,
  45, 93, 141, 189,
  6, 54, 102, 150,
  14, 62, 110, 158,
  22, 70, 118, 166,
  30, 78, 126, 174,
  38, 86, 134, 182,
  46, 94, 142, 190,
  7, 55, 103, 151,
  15, 63, 111, 159,
  23, 71, 119, 167,
  31, 79, 127, 175,
  39, 87, 135, 183,
  47, 95, 143, 191 ;
}
```
