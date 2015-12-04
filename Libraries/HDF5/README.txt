Contents:
(1) hdf5_test.f90: Create a random vector of dimension 100 
                   and write it to a hdf5 file
(2) Makefile

Compile and Run:
(1) Load required modules

source new-modules.sh
module load hdf5/1.8.12-fasrc08

(2) Compile

make

(3) Run

./hdf5_test.x
