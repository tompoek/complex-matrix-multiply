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
    // MPI SPLIT TASK: RANK0 FIRST HALF, RANK1 SECOND HALF, COPY BEFORE EXCHANGING BETWEEN NODES
    const int NperRank = N / world_size; // CONSTRAINT: N must be divisible by world_size
    // Rank0 first half, Rank1 second half >>>>
    const int jStart = (my_rank % world_size) * NperRank;
    int jEnd = jStart + NperRank;
    // int jOffset = NperRank;
    if (my_rank == world_size - 1) {
        jEnd = N;
        // jOffset = - NperRank * (world_size-1);
    }
    // Rank0 first half, Rank1 second half <<<<

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
            // _mm256_store_pd((double*)&C[i + (j+jOffset)*N], c_left); _mm256_store_pd((double*)&C[i + (j+jOffset+1)*N], c_right);
        }
    }
    }

    int NNperRank = N*N / world_size; // CONSTRAINT: N * N must be divisible by world_size
    const int iiCopyFrom = (my_rank % world_size) * NNperRank; // starting index to copy data from Matrix C's memory
    const int iiCopyTo = ((my_rank+1) % world_size) * NNperRank; // starting index to copy data into Matrix C's memory

    memcpy(C + iiCopyTo, C + iiCopyFrom, NNperRank * sizeof(floatType)); // copy data from one memory block to another based on node rank
    
    // #pragma omp parallel
    // {
    // #pragma omp for schedule(dynamic)
    // for (int ii = 0; ii < NNperRank; ii+=2) {
    //     _mm256_store_pd((double*)&C[ii + iiCopyTo], _mm256_load_pd((double*)&C[ii + iiCopyFrom]));
    // }
    // }

    MPI_Alltoall(C, NNperRank, MPI_DOUBLE_COMPLEX, C, NNperRank, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD); // exchange data between nodes in place










    // // Matrix multiplication using SIMD (AVX) with threading (OpenMP) and task-to-node distribution (MPI)
    // // For task distribution, split task in the middle with #nodes = 2
    // // MPI SPLIT TASK: RANK0 SECOND HALF, RANK1 FIRST HALF, CREATE COPY OF MATRIX C TO EXCHANGE BETWEEN NODES
    // const int NperRank = N / world_size; // CONSTRAINT: N must be divisible by world_size
    // // Rank0 second half, Rank1 first half >>>>
    // const int jStart = ((my_rank+1) % world_size) * NperRank;
    // int jEnd = jStart + NperRank;
    // if (my_rank == 0) {jEnd = N;}
    // // Rank0 second half, Rank1 first half <<<<
    // omp_set_num_threads(omp_get_max_threads());
    // #pragma omp parallel
    // {
    // #pragma omp for collapse(2) schedule(dynamic) // jobs threaded across i j
    // for (int i = 0; i < N; i+=2) {
    //     for (int j = jStart; j < jEnd; j+=2) { // if only one node, [jStart, jEnd]=[0, N]
    //         __m256d c_left = _mm256_setzero_pd(); __m256d c_right = _mm256_setzero_pd(); // defined within loop avoiding race conditions
    //         for (int k = 0; k < N; k+=2) {
    //             __m256d a_left = _mm256_load_pd((double*)&A[i + k*N]);
    //             __m256d a_right = _mm256_load_pd((double*)&A[i + (k+1)*N]);
    //             __m256d a_right_fliplane = _mm256_permute2f128_pd(a_right, a_right, 0b01); // flip across lane
    //             __m256d a_upper = _mm256_blend_pd(a_left, a_right_fliplane, 0b1100); // rearranged to what we want
    //             __m256d a_left_fliplane = _mm256_permute2f128_pd(a_left, a_left, 0b01); // flip across lane
    //             __m256d a_lower = _mm256_blend_pd(a_left_fliplane, a_right, 0b1100); // rearranged to what we want
    //             // C_LEFT >>>>>>
    //             __m256d b = _mm256_load_pd((double*)&B[k + j*N]); // b_left
    //             // Dot product >>>>>>
    //             __m256d a_r = _mm256_permute_pd(a_upper, 0b0000);
    //             __m256d a_i = _mm256_permute_pd(a_upper, 0b1111);
    //             __m256d b_r = _mm256_permute_pd(b, 0b0000);
    //             __m256d b_i = _mm256_permute_pd(b, 0b1111);
    //             __m256d rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             __m256d rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             __m256d c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             __m256d ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             __m256d ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             __m256d c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             __m256d c_upper = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             // Dot product >>>>>>
    //             a_r = _mm256_permute_pd(a_lower, 0b0000);
    //             a_i = _mm256_permute_pd(a_lower, 0b1111);
    //             b_r = _mm256_permute_pd(b, 0b0000);
    //             b_i = _mm256_permute_pd(b, 0b1111);
    //             rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             __m256d c_lower = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             c_left = _mm256_add_pd(c_left, _mm256_blend_pd(c_upper, c_lower, 0b1100));
    //             // C_LEFT <<<<<<
    //             // C_RIGHT <<<<<<
    //             b = _mm256_load_pd((double*)&B[k + (j+1)*N]); // b_right
    //             // Dot product >>>>>>
    //             a_r = _mm256_permute_pd(a_upper, 0b0000);
    //             a_i = _mm256_permute_pd(a_upper, 0b1111);
    //             b_r = _mm256_permute_pd(b, 0b0000);
    //             b_i = _mm256_permute_pd(b, 0b1111);
    //             rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             c_upper = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             // Dot product >>>>>>
    //             a_r = _mm256_permute_pd(a_lower, 0b0000);
    //             a_i = _mm256_permute_pd(a_lower, 0b1111);
    //             b_r = _mm256_permute_pd(b, 0b0000);
    //             b_i = _mm256_permute_pd(b, 0b1111);
    //             rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             c_lower = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             c_right = _mm256_add_pd(c_right, _mm256_blend_pd(c_upper, c_lower, 0b1100));
    //             // C_RIGHT <<<<<<
    //         }
    //         _mm256_store_pd((double*)&C[i + j*N], c_left); _mm256_store_pd((double*)&C[i + (j+1)*N], c_right);
    //         // _mm256_store_pd((double*)&C[i + (j+jOffset)*N], c_left); _mm256_store_pd((double*)&C[i + (j+jOffset+1)*N], c_right);
    //     }
    // }
    // }
    // int NNperRank = N*N / world_size; // CONSTRAINT: N * N must be divisible by world_size
    // floatType* C_copy = (floatType*)_mm_malloc(N*N*sizeof(floatType), 64);
    // MPI_Alltoall(C, NNperRank, MPI_DOUBLE_COMPLEX, C_copy, NNperRank, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD);
    // const int iiStart = (my_rank % world_size) * NNperRank;
    // const int iiCopyStart = ((my_rank+1) % world_size) * NNperRank;
    // #pragma omp parallel
    // {
    // #pragma omp for schedule(dynamic)
    // for (int ii = 0; ii < NNperRank; ii+=2) {
    //     _mm256_store_pd((double*)&C[ii+iiStart], _mm256_load_pd((double*)&C_copy[ii+iiCopyStart]));
    // }
    // }
    // _mm_free(C_copy);










    // // Matrix multiplication using SIMD (AVX) with threading (OpenMP) and task-to-node distribution (MPI)
    // // SPLIT TASK WITH CACHE LOCALITY
    // omp_set_num_threads(omp_get_max_threads());
    // #pragma omp parallel
    // {
    // #pragma omp for collapse(2) schedule(dynamic) // jobs threaded across i j
    // for (int i = 0; i < N; i+=2) {
    //     for (int jRank0 = 0; jRank0 < N; jRank0+=4) { // if only one node, [jStart, jEnd]=[0, N]
    //         int j = jRank0 + my_rank * 2;
    //         int jNextRank = jRank0 + ((my_rank+1) * 2) % world_size;
    //         __m256d c_left = _mm256_setzero_pd(); __m256d c_right = _mm256_setzero_pd(); // defined within loop avoiding race conditions
    //         for (int k = 0; k < N; k+=2) {
    //             __m256d a_left = _mm256_load_pd((double*)&A[i + k*N]);
    //             __m256d a_right = _mm256_load_pd((double*)&A[i + (k+1)*N]);
    //             __m256d a_right_fliplane = _mm256_permute2f128_pd(a_right, a_right, 0b01); // flip across lane
    //             __m256d a_upper = _mm256_blend_pd(a_left, a_right_fliplane, 0b1100); // rearranged to what we want
    //             __m256d a_left_fliplane = _mm256_permute2f128_pd(a_left, a_left, 0b01); // flip across lane
    //             __m256d a_lower = _mm256_blend_pd(a_left_fliplane, a_right, 0b1100); // rearranged to what we want
    //             // C_LEFT >>>>>>
    //             __m256d b = _mm256_load_pd((double*)&B[k + j*N]); // b_left
    //             // Dot product >>>>>>
    //             __m256d a_r = _mm256_permute_pd(a_upper, 0b0000);
    //             __m256d a_i = _mm256_permute_pd(a_upper, 0b1111);
    //             __m256d b_r = _mm256_permute_pd(b, 0b0000);
    //             __m256d b_i = _mm256_permute_pd(b, 0b1111);
    //             __m256d rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             __m256d rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             __m256d c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             __m256d ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             __m256d ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             __m256d c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             __m256d c_upper = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             // Dot product >>>>>>
    //             a_r = _mm256_permute_pd(a_lower, 0b0000);
    //             a_i = _mm256_permute_pd(a_lower, 0b1111);
    //             b_r = _mm256_permute_pd(b, 0b0000);
    //             b_i = _mm256_permute_pd(b, 0b1111);
    //             rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             __m256d c_lower = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             c_left = _mm256_add_pd(c_left, _mm256_blend_pd(c_upper, c_lower, 0b1100));
    //             // C_LEFT <<<<<<
    //             // C_RIGHT <<<<<<
    //             b = _mm256_load_pd((double*)&B[k + (j+1)*N]); // b_right
    //             // Dot product >>>>>>
    //             a_r = _mm256_permute_pd(a_upper, 0b0000);
    //             a_i = _mm256_permute_pd(a_upper, 0b1111);
    //             b_r = _mm256_permute_pd(b, 0b0000);
    //             b_i = _mm256_permute_pd(b, 0b1111);
    //             rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             c_upper = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             // Dot product >>>>>>
    //             a_r = _mm256_permute_pd(a_lower, 0b0000);
    //             a_i = _mm256_permute_pd(a_lower, 0b1111);
    //             b_r = _mm256_permute_pd(b, 0b0000);
    //             b_i = _mm256_permute_pd(b, 0b1111);
    //             rr_minus_ii = _mm256_sub_pd(_mm256_mul_pd(a_r,b_r), _mm256_mul_pd(a_i,b_i));
    //             rr_minus_ii_fliplane = _mm256_permute2f128_pd(rr_minus_ii, rr_minus_ii, 0b01);
    //             c_r = _mm256_add_pd(rr_minus_ii, rr_minus_ii_fliplane);
    //             ri_plus_ir = _mm256_add_pd(_mm256_mul_pd(a_r,b_i), _mm256_mul_pd(a_i,b_r));
    //             ri_plus_ir_fliplane = _mm256_permute2f128_pd(ri_plus_ir, ri_plus_ir, 0b01);
    //             c_i = _mm256_add_pd(ri_plus_ir, ri_plus_ir_fliplane);
    //             c_lower = _mm256_blend_pd(c_r, c_i, 0b1010);
    //             // Dot product <<<<<<
    //             c_right = _mm256_add_pd(c_right, _mm256_blend_pd(c_upper, c_lower, 0b1100));
    //             // C_RIGHT <<<<<<
    //         }
    //         _mm256_store_pd((double*)&C[i + j*N], c_left); _mm256_store_pd((double*)&C[i + (j+1)*N], c_right);
    //         _mm256_store_pd((double*)&C[i + jNextRank*N], c_left); _mm256_store_pd((double*)&C[i + (jNextRank+1)*N], c_right);
    //     }
    // }
    // }
    // // THE CODES ABOVE HASN'T BEEN TESTED, BECAUSE CODES BELOW WON'T WORK SINCE WE ONLY HAVE TWO NODES:
    // MPI_Alltoall(C, 2, MPI_DOUBLE_COMPLEX, C, 2, MPI_DOUBLE_COMPLEX, MPI_COMM_WORLD); // exchange data between nodes in place

}