#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = gfortran
F90CFLAGS   = -c -O2
INC         =  -I ${HDF5_INCLUDE}
LIBS        = -lhdf5 -lhdf5_fortran
PRO         = hdf5_test

OBJECTS = hdf5_test.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(INC) $(<F)

clean : 
	rm *.o *.x
