#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2 -openmp
COMPILER = icc
PRO         = omp_loop
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) -openmp

%.o : %.c
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x
