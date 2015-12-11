#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = gfortran
F90CFLAGS   = -c -O2
INC         =  -I ${FFTW_INCLUDE}
LIBS        = -lfftw3 -lm
PRO         = fftw_test

OBJECTS = fftw_test.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(INC) $(<F)

clean : 
	rm *.o *.x