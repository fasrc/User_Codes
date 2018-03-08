EXECS=communicator_mpi.x
MPICC?=mpicc

all: ${EXECS}

communicator_mpi.x: communicator_mpi.c
	${MPICC} -o communicator_mpi.x communicator_mpi.c

clean:
	rm -f ${EXECS}
