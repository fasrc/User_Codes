#==========================================================
# Make file
#==========================================================
CFLAGS   = -c -O2 -fopenmp
COMPILER = gfortran
PRO         = omp_mem_test
OBJECTS     = omp_mem_test.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -fopenmp

%.o : %.f90
	$(COMPILER) $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.out *.err
