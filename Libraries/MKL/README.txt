Contents:
(1) mkltest.f90: Create a random 100x100 and diagonalize it
(2) Makefile

Compile and Run:
(1) Load required modules

source new-modules.sh
module load intel/15.0.0-fasrc01
module load intel-mkl/11.0.0.079-fasrc02

(2) Compile

make

(3) Run

./mkltest.x
