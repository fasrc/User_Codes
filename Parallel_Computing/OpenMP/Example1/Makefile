#==========================================================
# Make file for omp_dot.cpp
#==========================================================
F90CFLAGS   = -c -O2 -fopenmp
F90COMPILER = g++
PRO         = omp_dot
OBJECTS     = $(PRO).o

LIBS        = -fopenmp

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.cpp
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm *.o *.x
