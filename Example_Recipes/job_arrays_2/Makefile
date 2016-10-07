#++++++++++++++++++++++++++++++++++++++++++++++++
# Makefile
#++++++++++++++++++++++++++++++++++++++++++++++++
.PHONY: all build clean

COMP     = g++
OPTIONS  = -O2
PRO      = main_program
EXE      = pro

all: build

build:
	$(COMP) -o $(EXE).x $(PRO).cpp $(OPTIONS)

clean:
	rm *.x

run:
	sbatch run.sbatch
