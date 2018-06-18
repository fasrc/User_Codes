#==========================================================================
# Makefile
#==========================================================================
F90COMPILER = mpifort
F90CFLAGS   = -c  -i8 -I${MKL_HOME}/include
PRO         = psgesv

MKLROOT = $(MKL_HOME)

LIBS = ${MKLROOT}/lib/intel64/libmkl_scalapack_ilp64.a -Wl,--start-group \
	${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
	${MKLROOT}/lib/intel64/libmkl_sequential.a \
	${MKLROOT}/lib/intel64/libmkl_core.a \
	${MKLROOT}/lib/intel64/libmkl_blacs_openmpi_ilp64.a -Wl,--end-group \
	 -lpthread -lm -ldl 

OBJECTS =  $(PRO).o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

%.o : %.F90
	$(F90COMPILER) $(F90CFLAGS) $(<F)


clean : 
	rm -rf *.o *.x
