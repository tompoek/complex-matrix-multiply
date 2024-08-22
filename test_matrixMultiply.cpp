#include <iostream>

//Alignment boundary for aligned mallocs (e.g. _mm_malloc). 64 bytes = cacheline
#define ALIGN 64

#include "Assignment1_GradeBot.h"
#include "matrixMultiply.h"

void matrixMultiplyExpectedResult(int N, floatType* A, floatType* B, floatType* C) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                double real = A[i + k * N].real() * B[k + j * N].real() - A[i + k * N].imag() * B[k + j * N].imag();
                double imag = A[i + k * N].real() * B[k + j * N].imag() + A[i + k * N].imag() * B[k + j * N].real();
                floatType complex(real, imag);
                C[i + j * N] += complex;
                // C[i + j * N] += A[i + k * N] * B[k + j * N]; // equivalent operation
            }
        }
    }
}

int main() {
    
    int N = 4;
    floatType* A = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    floatType* B = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    floatType* C = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    // memset(C, 0, N * N * sizeof(floatType));
    floatType* C_expected = (floatType*)_mm_malloc(N*N*sizeof(floatType), ALIGN);
    memset(C_expected, 0, N * N * sizeof(floatType));

    // Example matrix A
    // A[0] = floatType(1, 1);
    // A[1] = floatType(1, -1);
    // A[2] = floatType(-1, 1);
    // A[3] = floatType(-1, -1);
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
    matrixMultiply(N, A, B, C, args, argCount);

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
}