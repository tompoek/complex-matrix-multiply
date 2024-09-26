#include <iostream>

//Alignment boundary for aligned mallocs (e.g. _mm_malloc). 64 bytes = cacheline
#define ALIGN 64

#include "Assignment1_GradeBot.h"
#include "matrixMultiply.h"
#include "matrixMultiplyGPU.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>

void matrixMultiplyExpectedResult(int N, floatType* A, floatType* B, floatType* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double real = A[i + k * N].real() * B[k + j * N].real() - A[i + k * N].imag() * B[k + j * N].imag();
                double imag = A[i + k * N].real() * B[k + j * N].imag() + A[i + k * N].imag() * B[k + j * N].real();
                floatType complex(real, imag);
                C[i + j * N] += complex;
                // C[i + j * N] += A[i + k * N] * B[k + j * N]; // equivalent operation without splitting real/imag parts
            }
        }
    }
}

int main() {
    
    int N = 8;
    floatType* A = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    floatType* B = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    floatType* C = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    floatTypeCUDA* A_device;
    floatTypeCUDA* B_device;
    floatTypeCUDA* C_device;
    cudaMalloc(&A_device, N*N*sizeof(*A_device));
    cudaMalloc(&B_device, N*N*sizeof(*B_device));
    cudaMalloc(&C_device, N*N*sizeof(*C_device));
    memset(C, 0, N * N * sizeof(floatType));
    floatType* C_expected = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    memset(C_expected, 0, N * N * sizeof(floatType));

    // Example matrix A
    A[0] = floatType(0, 1);
    A[1] = floatType(2, 3);
    A[2] = floatType(4, 5);
    A[3] = floatType(6, 7);
    if (N >= 3) {
    A[4] = A[0] * floatType(2, 0);
    A[5] = A[1] * floatType(2, 0);
    A[6] = A[2] * floatType(2, 0);
    A[7] = A[3] * floatType(2, 0);
    A[8] = A[0] * floatType(3, 0);
    if (N >= 4) {
    A[9] = A[1] * floatType(3, 0);
    A[10] = A[2] * floatType(3, 0);
    A[11] = A[3] * floatType(3, 0);
    A[12] = A[0] * floatType(4, 0);
    A[13] = A[1] * floatType(4, 0);
    A[14] = A[2] * floatType(4, 0);
    A[15] = A[3] * floatType(4, 0);
    if (N >= 6) {
    A[16] = A[1] * floatType(5, 0);
    A[17] = A[2] * floatType(5, 0);
    A[18] = A[3] * floatType(5, 0);
    A[19] = A[0] * floatType(5, 0);
    A[20] = A[1] * floatType(6, 0);
    A[21] = A[2] * floatType(6, 0);
    A[22] = A[3] * floatType(6, 0);
    A[23] = A[1] * floatType(6, 0);
    A[24] = A[2] * floatType(7, 0);
    A[25] = A[3] * floatType(7, 0);
    A[26] = A[0] * floatType(7, 0);
    A[27] = A[1] * floatType(7, 0);
    A[28] = A[2] * floatType(8, 0);
    A[29] = A[3] * floatType(8, 0);
    A[30] = A[1] * floatType(8, 0);
    A[31] = A[2] * floatType(8, 0);
    A[32] = A[3] * floatType(9, 0);
    A[33] = A[0] * floatType(9, 0);
    A[34] = A[1] * floatType(9, 0);
    A[35] = A[2] * floatType(9, 0);
    if (N >= 8) {
    for (int i=36; i<64; i++) {A[i] = A[i-36];}
    }
    }
    }
    }
    // Print the matrix A
    std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << A[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    // Example matrix B
    B[0] = floatType(7, 6);
    B[1] = floatType(5, 4);
    B[2] = floatType(3, 2);
    B[3] = floatType(1, 0);
    if (N >= 3) {
    B[4] = B[0] * floatType(2, 0);
    B[5] = B[1] * floatType(2, 0);
    B[6] = B[2] * floatType(2, 0);
    B[7] = B[3] * floatType(2, 0);
    B[8] = B[0] * floatType(3, 0);
    if (N >= 4) {
    B[9] = B[1] * floatType(3, 0);
    B[10] = B[2] * floatType(3, 0);
    B[11] = B[3] * floatType(3, 0);
    B[12] = B[0] * floatType(4, 0);
    B[13] = B[1] * floatType(4, 0);
    B[14] = B[2] * floatType(4, 0);
    B[15] = B[3] * floatType(4, 0);
    if (N >= 6) {
    B[16] = B[1] * floatType(5, 0);
    B[17] = B[2] * floatType(5, 0);
    B[18] = B[3] * floatType(5, 0);
    B[19] = B[0] * floatType(5, 0);
    B[20] = B[1] * floatType(6, 0);
    B[21] = B[2] * floatType(6, 0);
    B[22] = B[3] * floatType(6, 0);
    B[23] = B[1] * floatType(6, 0);
    B[24] = B[2] * floatType(7, 0);
    B[25] = B[3] * floatType(7, 0);
    B[26] = B[0] * floatType(7, 0);
    B[27] = B[1] * floatType(7, 0);
    B[28] = B[2] * floatType(8, 0);
    B[29] = B[3] * floatType(8, 0);
    B[30] = B[1] * floatType(8, 0);
    B[31] = B[2] * floatType(8, 0);
    B[32] = B[3] * floatType(9, 0);
    B[33] = B[0] * floatType(9, 0);
    B[34] = B[1] * floatType(9, 0);
    B[35] = B[2] * floatType(9, 0);
    if (N >= 8) {
    for (int i=36; i<64; i++) {B[i] = B[i-36];}
    }
    }
    }
    }
    // Print the matrix B
    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << B[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    int* args = 0;
    int argCount = 0;
    // Copy matrices from host to device
    cudaMemcpy(A_device, A, N*N*sizeof(*A_device), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, N*N*sizeof(*B_device), cudaMemcpyHostToDevice);
    cudaMemcpy(C_device, C, N*N*sizeof(*C_device), cudaMemcpyHostToDevice);
    matrixMultiply_GPU(N, A_device, B_device, C_device, args, argCount);
    // Copy result matrix C from device back to host
    cudaMemcpy(C, C_device, N*N*sizeof(*C_device), cudaMemcpyDeviceToHost);

    // Print the result matrix C
    std::cout << "Matrix C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    // Perform matrix multiplication and store to C_expected
    matrixMultiplyExpectedResult(N, A, B, C_expected);
    // Print the matrix C_expected
    std::cout << "Matrix C_expected:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C_expected[i + j * N] << " ";
        }
        std::cout << std::endl;
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    _mm_free(C_expected);
    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);
}