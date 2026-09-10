MPICC      := mpicc
MANA_MPICC := mpicc_mana

CFLAGS := -std=c11 -O2 -Wall -Wextra -pedantic
LDLIBS := -lm

SRC := pi_mpi.c

TARGETS := pi_mpi.x pi_mpi_mana.x

.PHONY: all clean

all: $(TARGETS)

pi_mpi.x: $(SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

pi_mpi_mana.x: $(SRC)
	$(MANA_MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f $(TARGETS)
