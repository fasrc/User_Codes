Contents:

(1) fftw_test.f90: Perform 1D transform
(2) Makefile

Compile and Run:

(1) Load required modules, e.g.,

source new-modules.sh
module load gcc/4.8.2-fasrc01
module load openmpi/1.8.3-fasrc02
module load fftw/3.3.4-fasrc06 

(2) Compile

make

(3) Run

./fftw_test.x

Example output:
 
 Input array:
     in(           1 ) =  (  1.0000000000000000     ,  2.0000000000000000     )
     in(           2 ) =  (  2.0000000000000000     ,  3.0000000000000000     )
     in(           3 ) =  (  3.0000000000000000     ,  4.0000000000000000     )
     in(           4 ) =  (  4.0000000000000000     ,  5.0000000000000000     )
 Output array:
     out(           1 ) =  (  10.000000000000000     ,  14.000000000000000     )
     out(           2 ) =  ( -4.0000000000000000     ,  0.0000000000000000     )
     out(           3 ) =  ( -2.0000000000000000     , -2.0000000000000000     )
     out(           4 ) =  (  0.0000000000000000     , -4.0000000000000000     )
 Output array after inverse FFT:
                4  * in(           1 ) =  (  4.0000000000000000     ,  8.0000000000000000     )
                4  * in(           2 ) =  (  16.000000000000000     ,  20.000000000000000     )
                4  * in(           3 ) =  (  12.000000000000000     ,  16.000000000000000     )
                4  * in(           4 ) =  (  8.0000000000000000     ,  12.000000000000000     )
