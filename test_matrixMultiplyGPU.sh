set -o errexit

rm -f test_matrixMultiplyGPU.o matrixMultiplyGPU.o 
nvcc -o test_matrixMultiplyGPU.o test_matrixMultiplyGPU.cpp matrixMultiplyGPU.cu -O2 --gpu-architecture=sm_35 -Wno-deprecated-gpu-targets 
./test_matrixMultiplyGPU.o
