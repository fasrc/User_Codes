EXECS=prime_mpi
MPICC?=mpicc

all: ${EXECS}

prime_mpi: prime_mpi.c
	${MPICC} -o prime_mpi prime_mpi.c

clean:
	rm -f ${EXECS}
