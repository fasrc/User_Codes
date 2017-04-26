EXECS=poisson_mpi
MPICC?=mpicc

all: ${EXECS}

poisson_mpi: poisson_mpi.c
	${MPICC} -o poisson_mpi poisson_mpi.c

clean:
	rm -f ${EXECS}
