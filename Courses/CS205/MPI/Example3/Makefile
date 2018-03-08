EXECS=mmult.x
MPICC?=mpicc

all: ${EXECS}

mmult.x: mmult.c
	${MPICC} -o mmult.x mmult.c

clean:
	rm -f ${EXECS}
