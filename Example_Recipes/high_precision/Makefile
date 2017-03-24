#==========================================================
# Makefile
#==========================================================
F90COMPILER = ifort
COMPILER    = icc
LD          = ifort
F90CFLAGS   = -O2 -c -fast -heap-arrays -assume nounderscore
CFLAGS      = -O2 -c 
LIBS        = -lmpfr -lgmp

# Program..................................................
PRO         = mpint_test_v2

# Object files.............................................
OBJECTS = mpfuna.o      \
          mpfunf.o      \
          mpfung1.o     \
          mpmodule.o    \
          mpinterface.o \
          secondu.o     \
          mpint_lib.o   \
          ${PRO}.o

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

%.o : %.c
	$(COMPILER) $(CFLAGS) $(<F)

${PRO}.x : $(OBJECTS)
	$(LD) -o ${PRO}.x $(OBJECTS) $(LIBS)

clean : 
	rm -rf *.o *.mod *.x *~
