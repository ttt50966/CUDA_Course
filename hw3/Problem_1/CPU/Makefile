CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v

BIN_CPU := laplace_cpu
LIB     :=

all: $(BIN_CPU)

$(BIN_CPU): laplace_cpu.cu
	$(NVCC) -o $(BIN_CPU) $(NVFLAGS) laplace_cpu.cu $(LIB)
clean:
	rm -f $(BIN_CPU)
