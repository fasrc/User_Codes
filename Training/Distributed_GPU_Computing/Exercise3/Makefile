# Compiler
NVCC = nvcc

# Target
TARGET = nccl_pi_mc.x

# Source
SRC = nccl_pi_mc.cu

# Libraries
LIBS = -lnccl

# Default rule
all: $(TARGET)

# Build rule
$(TARGET): $(SRC)
	$(NVCC) -o $@ $^ $(LIBS)

# Clean rule
clean:
	rm -f $(TARGET)
