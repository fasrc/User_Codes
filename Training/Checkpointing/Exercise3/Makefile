CC      := gcc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -pedantic
LDLIBS  := -lm

PROGRAM := long_run

.PHONY: all clean

all: $(PROGRAM)

$(PROGRAM): long_run.c
	$(CC) $(CFLAGS) -o $@.x $< $(LDLIBS)

clean:
	rm -f $(PROGRAM).x
	rm -f *.out *.err
	rm -f ckpt_*.dmtcp
	rm -f dmtcp_restart_*.sh
	rm -f coordinator.port
	rm -rf checkpoints/

