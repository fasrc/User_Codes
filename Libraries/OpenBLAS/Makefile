#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = gfortran
F90CFLAGS   = -c -O2
LIBS        = -lopenblas
PRO         = openblas_test

OBJECTS = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(INC) $(<F)

clean : 
	rm -rf *.o *.x
