#==========================================================
# Make file for mpi_IO_test1.f90
#==========================================================
F90CFLAGS   = -c -O2
F90COMPILER = mpif90
PRO         = mpi_IO_test1
OBJECTS     = $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm -rf *.o *.x *.mod *.lvec
