EXECS=mmult
MPICC?=mpicc

all: ${EXECS}

mmult: mmult.c
	${MPICC} -o mmult mmult.c

clean:
	rm -f ${EXECS}
