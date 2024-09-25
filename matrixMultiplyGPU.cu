#include "matrixMultiplyGPU.cuh"

__host__ void matrixMultiply_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int* flags, int flagCount)
{
    int numThreadsPerBlock = 8 /*different #threads per block settings yield different performance*/;
    int numBlocks = N / numThreadsPerBlock /*make sure the total #threads = matrix size N*/;
    matrixMultiplyKernel_GPU<<<dim3(numBlocks,numBlocks), dim3(numThreadsPerBlock,numThreadsPerBlock)>>>/*call kernel function*/(N, A, B, C, 0, 0, 0);
}


__global__ void matrixMultiplyKernel_GPU(int N, const floatTypeCUDA* A, const floatTypeCUDA* B, floatTypeCUDA* C, int flag0, int flag1, int flag2)
{
    int i /*distribute i across x axis in 2D threading*/ = blockIdx.x * blockDim.x + threadIdx.x;
    int j /*distribute j across y axis in 2D threading*/ = blockIdx.y * blockDim.y + threadIdx.y;
    cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);

    for (int k = 0; k < N; k += 1) /*each thread loops over k to perform dot product*/ {
        sum = cuCadd(sum, cuCmul(A[i + k * N], B[k + j * N]));
    }

    C[i + j * N] /*each thread sums up dot product to the C element of its unique i and j*/ = sum;
}
