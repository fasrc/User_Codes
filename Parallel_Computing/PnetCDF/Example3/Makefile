#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpiifort
F90CFLAGS   = -c -O2
LIBS        = -lpnetcdf
PRO         = block_cyclic

OBJECTS =  utils.o     \
           block_cyclic.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

%.o : %.F90
	$(F90COMPILER) $(F90CFLAGS) $(<F)


clean : 
	rm -rf *.o *.x
