CC      := gcc
CFLAGS  := -std=c11 -O2 -Wall -Wextra -pedantic
LDLIBS  := -lm

PROGRAMS := \
	pi_naive \
	pi_checkpoint \
	pi_signal \
	pi_requeue


.PHONY: all clean

all: $(PROGRAMS)


pi_naive: pi_naive.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)


pi_checkpoint: pi_checkpoint.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)


pi_signal: pi_signal.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)


pi_requeue: pi_requeue.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)


clean:
	rm -f $(PROGRAMS)
	rm -f checkpoints/*.bin
	rm -f checkpoints/*.bin.tmp
