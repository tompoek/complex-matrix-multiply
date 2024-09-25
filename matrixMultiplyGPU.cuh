#include <vector_types.h>//allows the use of composite types, e.g. float2, float3, int3, etc.
#include <cuComplex.h>
#include "Assignment1_GradeBot.h"
/**
* @brief Implements an NxN matrix multiply C=A*B
*
* @param[in] N : dimension of square matrix (NxN)
* @param[in] A : pointer to input NxN matrix
* @param[in] B : pointer to input NxN matrix
* @param[out] C : pointer to output NxN matrix
* @param[in] flags : pointer to array of integers which can be used for debugging and performance tweaks. Optional. If unused, set to zero
* @param[in] flagCount : the length of the flags array
* @return void
* */
__host__ void matrixMultiply_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int* flags, int flagCount);

//The kernel (device code) parameters have been setup almost the same as the host code, except the flags are passed in individually rather than as a pointer. This is done just so you don't have to copy the parameters to GPU memory first, you'll be able to pass in up to 3 on the function call.
__global__ void matrixMultiplyKernel_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int flag0, int flag1, int flag2);