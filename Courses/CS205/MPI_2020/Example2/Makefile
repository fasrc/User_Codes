#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2
COMPILER = mpicxx
PRO         = mpi_dot
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x
