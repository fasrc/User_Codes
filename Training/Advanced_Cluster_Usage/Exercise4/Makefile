#==========================================================
# Make file
#==========================================================
F90CFLAGS   = -c -O2
F90COMPILER = mpif90
PRO         = pi_monte_carlo
OBJECTS     = pi_monte_carlo.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm -f *.o *.x *.out *.err

