#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2 -fopenmp
COMPILER = gcc
PRO         = omp_mm
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -fopenmp

%.o : %.c
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x *.out *.err *.dat
