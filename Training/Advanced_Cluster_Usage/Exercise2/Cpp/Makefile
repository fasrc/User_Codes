#==========================================================
# Make file
#==========================================================
F90CFLAGS   = -c -O2 -qopenmp
F90COMPILER = icpx
PRO         = planczos
OBJECTS     = planczos2.o external_libs.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) -qopenmp

%.o : %.cpp
	$(F90COMPILER) $(F90CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.mod *.lvec *.out *.err
