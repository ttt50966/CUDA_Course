CUDIR     := /usr/local/nvidia

CC      := gcc
CFLAGS  := -O2 -Wall -g
CPP     := g++
CFLAGS  := -O2 -Wall -g
NVCC    := $(CUDIR)/bin/nvcc
NVFLAGS := -O2 -I$(CUDIR)/include -m64 -arch=compute_52 -code=sm_52 -Xptxas -v

BIN     := vecAdd
LIB     :=

all: $(BIN)

$(BIN): vecAdd.cu
	$(NVCC) -o $(BIN) $(NVFLAGS) vecAdd.cu $(LIB)

clean:
	rm -f $(BIN)
