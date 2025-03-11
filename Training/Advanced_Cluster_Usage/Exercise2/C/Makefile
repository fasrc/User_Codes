#==========================================================
# Make file
#==========================================================
CFLAGS   = -c -O2 -qopenmp
COMPILER = icx
PRO         = planczos
OBJECTS     = planczos2.o external_libs.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -qopenmp

%.o : %.c
	$(COMPILER) $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.mod *.lvec *.out *.err
