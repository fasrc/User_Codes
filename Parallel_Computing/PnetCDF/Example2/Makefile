#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpiifort
F90CFLAGS   = -c -O2
LIBS        = -lpnetcdf
PRO         = column_wise

OBJECTS =  utils.o     \
           column_wise.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

%.o : %.F90
	$(F90COMPILER) $(F90CFLAGS) $(<F)


clean : 
	rm -rf *.o *.x
