#=======================================================================
# Make file for omp_diag.f90
#=======================================================================
F90CFLAGS   = -c -O3
F90COMPILER = ifort
PRO         = matrix_diag
OBJECTS     = matrix_diag.o

LIBMKL = -L $(MKLROOT)/lib/intel64/ -lmkl_blas95_lp64 -lmkl_lapack95_lp64 \
 -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group \
-lpthread
#LIBMKL = -lblas -llapack

LINK_MKL = $(LIBMKL)

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS) $(LINK_MKL)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean :
	rm -fr *.o *.x
