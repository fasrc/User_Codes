EXECS=communicator_mpi
MPICC?=mpicc

all: ${EXECS}

communicator_mpi: communicator_mpi.c
	${MPICC} -o communicator_mpi communicator_mpi.c

clean:
	rm -f ${EXECS}
