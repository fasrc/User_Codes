#==========================================================
# Make file
#==========================================================
CFLAGS   = -c -O2
COMPILER = g++
PRO         = mem_test
OBJECTS     = mem_test.o

# Default target
all: ${PRO}.x

${PRO}.x : $(OBJECTS)
	$(COMPILER) -o ${PRO}.x $(OBJECTS)

%.o : %.cpp
	$(COMPILER) $(CFLAGS) $<

.PHONY: clean
clean:
	rm -f *.o *.x *.out *.err
