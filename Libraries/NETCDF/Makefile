#==========================================================
# Makefile
#==========================================================
F90COMPILER = gfortran
F90CFLAGS   = -c -O2
INC         =  -I ${NETCDF_INCLUDE}
LIBS        = -lnetcdf -lnetcdff
PRO         = netcdf_test

OBJECTS = $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(INC) $(<F)

clean : 
	rm *.o *.x
