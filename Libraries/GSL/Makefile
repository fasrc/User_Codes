#=======================================================================
# Make file for gsl_int_test.c
#=======================================================================
CFLAGS   = -c -O2
COMPILER = gcc
PRO         = gsl_int_test
OBJECTS     = gsl_int_test.o

LINK_GSL = -lm -lgsl -lgslcblas

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS) $(LINK_GSL)

%.o : %.c
	$(COMPILER) $(CFLAGS) $(<F)

clean : 
	rm -rf *.o *.x *.mod
