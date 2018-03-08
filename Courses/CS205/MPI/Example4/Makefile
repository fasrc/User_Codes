EXECS=poisson_mpi.x
MPICC?=mpicc

all: ${EXECS}

poisson_mpi.x: poisson_mpi.c
	${MPICC} -o poisson_mpi.x poisson_mpi.c

clean:
	rm -f ${EXECS}
