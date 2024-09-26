# set some defaults.  ?= means don't override the value if it was set already
MPI_INC = /usr/include/openmpi-x86_64
MPI_LIB = /usr/lib64/openmpi/lib

CUDA_INC = /opt/local/stow/cuda-11.1/include
CUDA_LIB = /opt/local/stow/cuda-11.1/lib64

MPICXX?=g++
CXX?=g++
CXXFLAGS?=-std=c++11 -O2 -mavx -fopenmp -I$(MPI_INC) -I$(CUDA_INC)
NVCC?=nvcc
NVFLAGS?=-O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets 

# all targets
TARGETS = Assignment1_GradeBot

# The first rule in the Makefile is the default target that is made if 'make' is invoked with
# no parameters.  'all' is a dummy target that will make everything
default : all

## Dependencies



# all targets depend on the helper programs (MPI ENABLED ON CLUSTERS)
# $(TARGETS) : matrixMultiply.h matrixMultiply.o matrixMultiplyGPU.cuh matrixMultiplyGPU.o matrixMultiplyMPI.h matrixMultiplyMPI.o
# LIBS_Assignment1_GradeBot = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -ldl -L$(CUDA_LIB) -lcudart -lcublas -L$(MPI_LIB) -lmpi_cxx -lmpi

# CXXFLAGS_Assignment1_GradeBot =

# matrixMultiply.o : matrixMultiply.h
# matrixMultiplyGPU.o : matrixMultiplyGPU.cuh
# matrixMultiplyMPI.o : matrixMultiplyMPI.h
# Assignment1_GradeBot.o : Assignment1_GradeBot.h

# all targets depend on the helper programs (MPI DISABLED LOCALLY)
$(TARGETS) : matrixMultiply.h matrixMultiply.o matrixMultiplyGPU.cuh matrixMultiplyGPU.o
LIBS_Assignment1_GradeBot = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lm -ldl -L$(CUDA_LIB) -lcudart -lcublas -L$(MPI_LIB)

CXXFLAGS_Assignment1_GradeBot =

matrixMultiply.o : matrixMultiply.h
matrixMultiplyGPU.o : matrixMultiplyGPU.cuh
Assignment1_GradeBot.o : Assignment1_GradeBot.h




# wildcard rules
#%MPI.o : %MPI.cpp
#	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

#%MPI : %MPI.cpp
#	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%GPU.o : %GPU.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$(basename $<)) -c $< -o $@

% : %.cu
	$(NVCC) $(NVFLAGS) $(NVFLAGS_$@) $(filter %.o %.cu, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

%.o : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CFLAGS_$(basename $<)) -c $< -o $@

% : %.cpp
	$(MPICXX) $(CXXFLAGS) $(CXXFLAGS_$@) $(filter %.o %.cpp, $^) $(LDFLAGS) $(LIBS_$@) $(LIB) -o $@

all : $(TARGETS)

clean:
	rm -f $(TARGETS) *.o

.PHONY: clean default all
