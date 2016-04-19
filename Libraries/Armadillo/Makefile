#=======================================================================
# Make file for armadillo_test.cpp
#=======================================================================
CPPFLAGS    = -c -O2
CPPCOMPILER = g++
PRO         = armadillo_test
OBJECTS     = ${PRO}.o

LIBS = -larmadillo

${PRO}.x : $(OBJECTS)
	$(CPPCOMPILER) -o ${PRO}.x $(OBJECTS) $(LIBS)

%.o : %.cpp
	$(CPPCOMPILER) $(CPPFLAGS) $(<F)

clean : 
	rm *.o *.x
