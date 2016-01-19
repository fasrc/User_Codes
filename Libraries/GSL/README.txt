Contents:
(1) gsl_int_test.c: Solves the integral \int_0^1 x^{-1/2} log(x) dx = -4
(2) Makefile

Compile and Run:
(1) Load required modules

source new-modules.sh
module load gsl/1.16-fasrc02

(2) Compile

make

(3) Run

./gsl_int_test.x

(4) Example output

result          = -4.000000000000085265
exact result    = -4.000000000000000000
estimated error =  0.000000000000135447
actual error    = -0.000000000000085265
intervals       = 8
