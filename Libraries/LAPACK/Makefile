#=======================================================================
# Make file for lapack_test.f90
#=======================================================================
F90CFLAGS   = -c -O2
F90COMPILER = gfortran
PRO         = lapack_test
OBJECTS     = ${PRO}.o

LIBS = -lblas -llapack

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm *.o *.x
