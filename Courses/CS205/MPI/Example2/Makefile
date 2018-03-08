EXECS=prime_mpi.x
MPICC?=mpicc

all: ${EXECS}

prime_mpi.x: prime_mpi.c
	${MPICC} -o prime_mpi.x prime_mpi.c

clean:
	rm -f ${EXECS}
