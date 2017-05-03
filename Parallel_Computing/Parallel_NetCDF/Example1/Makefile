#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpiifort
F90CFLAGS   = -c -O2
LIBS        = -lnetcdf -lnetcdff
PRO         = f90tst_parallel

OBJECTS = $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm -rf *.o *.x
