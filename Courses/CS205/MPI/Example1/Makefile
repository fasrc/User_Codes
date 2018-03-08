EXECS=mpi_hello_world.x
MPICC?=mpicc

all: ${EXECS}

mpi_hello_world.x: mpi_hello_world.c
	${MPICC} -o mpi_hello_world.x mpi_hello_world.c

clean:
	rm -f ${EXECS}
