.PHONY: all build clean

COMP     = mex
PRO      = bessel_test
LINK_GSL = -L$(GSL_LIB) -lgsl -lgslcblas

all: build

build:
	$(COMP) $(PRO).cpp $(LINK_GSL)

clean:
	rm *.mexa64

 
