# Matrix Multiplication on Complex Numbers

## CPU version (in one node): 

* SIMD (Single Instruction Multiple Data) implemented via AVX (Advanced Vector eXtension),
* threaded across multiple cores using OpenMP (Open Multi Processing).
* To run locally and test correctness: ./test_matrixMultiply.sh

### The math behind AVX: (refer to matrixMultiply.cpp) 

![image](https://github.com/user-attachments/assets/d6344888-dd06-4814-9e74-2124410afd9b)

## CPU version (split by two nodes): 

* using MPI (Message Passing Interface). 
* Local testing not supported (need at least two nodes). 

### The split design: (refer to matrixMultiplyMPI.cpp) 

![image](https://github.com/user-attachments/assets/3f4f868c-4d12-4c05-b68b-55e00d07fb6f)

![image](https://github.com/user-attachments/assets/43dbdcbd-3f95-4d0e-aff4-77f7e04877e2)

## GPU version (in one node): 

* SIMT (Single Instruction Multiple Threads) implemented via CUDA, 
* split threads in x and y dimensions, for i and j indices, respectively,
* each thread computes the output element of unique i and j, with no race conditions. 
* To run locally and test correctness: ./test_matrixMultiplyGPU.sh

### The threading design: (refer to matrixMultiplyGPU.cpp) 

![image](https://github.com/user-attachments/assets/13283dd3-8871-44dd-ab94-b85f94ce6b68)

