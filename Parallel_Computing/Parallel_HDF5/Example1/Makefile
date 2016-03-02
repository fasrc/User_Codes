#==========================================================================
# Makefile for parallel_hdf5.f90
#==========================================================================
F90COMPILER = mpif90
F90CFLAGS   = -c -O2
LIBS        = -lhdf5 -lhdf5_fortran
PRO         = parallel_hdf5

OBJECTS = $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm *.o *.mod *.x
