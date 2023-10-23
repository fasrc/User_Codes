#======================================================================
# Make file for omp_diag.f90
#=======================================================================
F90CFLAGS   = -c -O2 -qopenmp
F90COMPILER = ifx
PRO         = omp_diag
OBJECTS     = omp_diag.o

LIBMKL = -L $(MKLROOT)/lib/intel64/ -lmkl_blas95_lp64 -lmkl_lapack95_lp64 \
 -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group \
-lpthread

LINK_MKL = $(LIBMKL)

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LINK_MKL) -qopenmp

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean :
	rm -f *.o *.x
