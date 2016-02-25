#=======================================================================
# Make file for blas_test.f90 
#=======================================================================
CFLAGS   = -c -O2
COMPILER = gfortran
PRO         = blas_test
OBJECTS     = blas_test.o

LINK_BLAS = -lblas

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) $(LINK_BLAS)

%.o : %.f90
	$(COMPILER) $(CFLAGS) $(<F)

clean : 
	rm *.o *.x
