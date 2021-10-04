#==========================================================
# Make file
#==========================================================
F90CFLAGS   = -c -w -fallow-argument-mismatch -O2
F90COMPILER = mpifort
PRO         = planczos
OBJECTS     = planczos2.o \
              external_libs.o

${PRO}.x : $(OBJECTS)
	$(F90COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.f90
	$(F90COMPILER) $(F90CFLAGS) $(<F)

clean : 
	rm -f *.o *.x *.mod


clean-output :
	rm -f planczos.err planczos.out planczos.dat lanczosvector.lvec
