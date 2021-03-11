#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2
COMPILER = mpicxx
PRO         = mpi_hello
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x

clean-output :
	rm -f mpi_hello.err mpi_hello.out mpi_hello.dat
