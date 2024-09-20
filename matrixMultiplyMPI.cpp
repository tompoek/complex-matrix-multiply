#include "matrixMultiplyMPI.h"

void matrixMultiply_MPI(int N, const floatType* A, const floatType* B, floatType* C, int* flags, int flagCount)
{

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Status status;

    // Matrix multiplication using SIMD (AVX) with threading (OpenMP) and task-to-node distribution (MPI)
    // For task distribution, split task in the middle with #nodes = 2
    const int NperRank = N / world_size; // CONSTRAINT: N must be divisible by world_size
    const int jStart = ((my_rank+1) /*Rank0 second half, Rank1 first half*/ % world_size) * NperRank;
    int jEnd = jStart + NperRank;

    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(dynamic) // jobs threaded across i j
    for (int i = 0; i < N; i+=2) {
        for (int j = jStart; j < jEnd; j+=2) { // if only one node, [jStart, jEnd]=[0, N]
            __m256d c_left = _mm256_setzero_pd(); __m256d c_right = _mm256_setzero_pd(); // defined within loop avoiding race conditions
            for (int k = 0; k < N; k+=2) {
                __m256d a_left = _mm256_load_pd((double*)&A[i + k*N]);
                __m256d a_right = _mm256_load_pd((double*)&A[i + (k+1)*N]);
                __m256d a_right_fliplane = _mm256_permute2f128_pd(a_right, a_right, 0b01); // flip across lane
                __m256d a_upper = _mm256_blend_pd(a_left, a_right_fliplane, 0b1100); // rearranged to what we want
                __m256d a_left_fliplane = _mm256_permute2f128_pd(a_left, a_left, 0b01); // flip across lane
                __m256d a_lower = _mm256_blend_pd(a_left_fliplane, a_right, 0b1100); // rearranged to what we want
                // C_LEFT >>>>>>
                __m256d b = _mm256_load_pd((double*)&B[k + j*N]); // b_left
                // Dot product >>>>>>
                __m256d a_r = _mm256_permute_pd(a_upper, 0b0000);
                __m256d a_i = _mm256_permute_pd(a_upper, 0b1111);
                __m256d b_r = _mm256_permute_pd(b, 0b0000);
                __m256d b_i = _mm256_permute_pd(b, 0b1111);
                __m256d rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
                __m256d rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
                __m256d c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
                __m256d ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
                __m256d ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
                __m256d c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
                __m256d c_upper = _mm256_blend_pd(c_r, c_i, 0b1010);
                // Dot product <<<<<<
                // Dot product >>>>>>
                a_r = _mm256_permute_pd(a_lower, 0b0000);
                a_i = _mm256_permute_pd(a_lower, 0b1111);
                b_r = _mm256_permute_pd(b, 0b0000);
                b_i = _mm256_permute_pd(b, 0b1111);
                rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
                rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
                c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
                ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
                ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
                c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
                __m256d c_lower = _mm256_blend_pd(c_r, c_i, 0b1010);
                // Dot product <<<<<<
                c_left = _mm256_add_pd(c_left, _mm256_blend_pd(c_upper, c_lower, 0b1100));
                // C_LEFT <<<<<<
                // C_RIGHT <<<<<<
                b = _mm256_load_pd((double*)&B[k + (j+1)*N]); // b_right
                // Dot product >>>>>>
                a_r = _mm256_permute_pd(a_upper, 0b0000);
                a_i = _mm256_permute_pd(a_upper, 0b1111);
                b_r = _mm256_permute_pd(b, 0b0000);
                b_i = _mm256_permute_pd(b, 0b1111);
                rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
                rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
                c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
                ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
                ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
                c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
                c_upper = _mm256_blend_pd(c_r, c_i, 0b1010);
                // Dot product <<<<<<
                // Dot product >>>>>>
                a_r = _mm256_permute_pd(a_lower, 0b0000);
                a_i = _mm256_permute_pd(a_lower, 0b1111);
                b_r = _mm256_permute_pd(b, 0b0000);
                b_i = _mm256_permute_pd(b, 0b1111);
                rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
                rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
                c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
                ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
                ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
                c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
                c_lower = _mm256_blend_pd(c_r, c_i, 0b1010);
                // Dot product <<<<<<
                c_right = _mm256_add_pd(c_right, _mm256_blend_pd(c_upper, c_lower, 0b1100));
                // C_RIGHT <<<<<<
            }
            _mm256_store_pd((double*)&C[i + j*N], c_left); _mm256_store_pd((double*)&C[i + (j+1)*N], c_right);
        }
    }
    }

    int NNperRank = N*N / world_size; // CONSTRAINT: N * N must be divisible by world_size
    floatType* C_copy = (floatType*)_mm_malloc(N*N*sizeof(floatType), 64);
    const int iiCopyFrom /*starting index of valid data in Matrix C*/ = (my_rank % world_size) * NNperRank;
    const int iiCopyTo /*starting index of valid data in Matrix C_copy*/ = ((my_rank+1) % world_size) * NNperRank;

    MPI_Alltoall(C, NNperRank, MPI_DOUBLE_COMPLEX, C_copy, NNperRank, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD); // scatter + gather to C_copy
    #pragma omp parallel
    {
    #pragma omp for schedule(dynamic)
    for (int ii = 0; ii < NNperRank; ii += 2) { // load results from C_copy and store them back to C
        _mm256_store_pd((double*)&C[iiCopyFrom + ii], _mm256_load_pd((double*)&C_copy[iiCopyTo + ii]));
    }
    }

}