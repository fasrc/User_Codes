# Compiler
NVCC = nvcc

# Target
TARGET = nccl_mpi_pi_mc.x

# Source
SRC = nccl_mpi_pi_mc.cu

# Flags
CXXFLAGS = -O3 -std=c++17
LIBS = -lnccl -lmpi

# Default rule
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
