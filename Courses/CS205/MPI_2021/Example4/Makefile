#=================================================
# Makefile
#=================================================
CFLAGS   = -c -O2
COMPILER = mpicc
PRO         = mmult
OBJECTS     = ${PRO}.o

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.c
	$(COMPILER) $(CFLAGS) $(<F)

clean :
	rm -fr *.o *.x

clean-output :
	rm -f mmult.err mmult.out mmult.dat
