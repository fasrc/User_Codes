#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2
COMPILER = mpicxx
PRO         = mpi_pi
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x

clean-output :
	rm -f mpi_pi.err mpi_pi.out mpi_pi.dat
