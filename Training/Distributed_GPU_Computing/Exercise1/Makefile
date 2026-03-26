# Compiler
NVCC        = nvcc

# Use MPI C++ compiler as host compiler
MPICXX      = mpicxx

# Target executable
TARGET      = mpi_cuda_pi.x

# Source file
SRC         = mpi_cuda_pi.cu

# Compiler flags
NVCC_FLAGS  = -O3 -ccbin $(MPICXX)

# Default target
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Clean rule
clean:
	rm -f $(TARGET) *.err *.out
