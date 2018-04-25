CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O3 -Wall -g
CPP     := g++
CFLAGS  := -O3 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O3 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v

BIN_TEX := laplaceTex
LIB     :=

all: $(BIN_TEX)

$(BIN_TEX): laplaceTex.cu
	$(NVCC) -o $(BIN_TEX) $(NVFLAGS) laplaceTex.cu $(LIB)
clean:
	rm -f $(BIN_TEX)
