# Makefile for Hybrid MPI + CUDA + OpenMP program

# Compiler
NVCC = nvcc

# Output binary name
TARGET = mpi_cuda.x

# Source file
SRC = mpi_cuda.cu

# Compiler flags
CXXFLAGS = -Xcompiler -fopenmp

# Include and library paths (customizable)
INCLUDES = -I$(MPI_INCLUDE)
LIBS     = -L$(MPI_LIB) -lmpi -lgomp

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET) $(INCLUDES) $(LIBS)

# Clean target
clean:
	rm -f $(TARGET)

# Phony targets
.PHONY: all clean

