# Set your C++ compiler
CXX := g++
# Set the CUDA compiler
NVCC := nvcc

# --- IMPORTANT: SET THESE PATHS ---
# Set the path to your downloaded LibTorch directory
LIBTORCH_PATH := /home/tlh24/venv/lib/python3.13/site-packages/torch
# Set the path to your CUDA installation
CUDA_INCLUDE_PATH := /usr/include/cuda
# CUDA_LIB_PATH := /usr/lib/cuda
CUDA_LIB_PATH := /home/tlh24/venv/lib/python3.13/site-packages/nvidia/cuda_runtime/lib/

# --- Compiler and Linker Flags ---
# C++ flags
CXXFLAGS := -g -Wall -std=c++17
# Include directories for C++ compiler
INCLUDES := -I$(LIBTORCH_PATH)/include -I$(LIBTORCH_PATH)/include/torch/csrc/api/include -I$(CUDA_INCLUDE_PATH)/include -I/usr/include/python3.13/
# Library directories
LDFLAGS := -L$(LIBTORCH_PATH)/lib -L$(CUDA_LIB_PATH) -Wl,-rpath,$(LIBTORCH_PATH)
# Libraries to link against
LIBS := -ltorch -ltorch_cpu -ltorch_cuda -lc10 $(CUDA_LIB_PATH)libcudart.so.12 -lpython3.13

# NVCC flags
NVCCFLAGS := -g -G -std=c++17 --compiler-options "-fPIC" -gencode arch=compute_89,code=sm_89 # was 89 for 4090, 86 for the 3090

PYTHON_CFLAGS   := $(shell python3-config --cflags)
PYTHON_LDFLAGS  := $(shell python3-config --ldflags)

# --- Source Files and Executable Name ---
TARGET := test_harness
CPP_SRCS := test_harness.cpp
CU_SRCS := forward.cu
OBJS := $(CPP_SRCS:.cpp=.o) $(CU_SRCS:.cu=.o)

# --- Build Rules ---
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS) $(LIBS) $(PYTHON_LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PYTHON_CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf *.o

# to run: (e.g.)
# LD_LIBRARY_PATH=/home/tlh24/venv/lib/python3.13/site-packages/torch/lib cuda-gdb ./test_harness
# for python install:
# pip install --no-build-isolation . -v
