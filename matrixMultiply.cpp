#include "matrixMultiply.h"

void matrixMultiply(int N, const floatType* A, const floatType* B, floatType* C, int* args, int argCount)
{

    // // Matrix multiplication using scalar * vector (AVX), NO THREADING
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         __m256d c = _mm256_setzero_pd();
    //         for (int k = 0; k < N; k+=2) {
    //             floatType a_left = A[i + k * N]; // load a left
    //             floatType a_right = A[i + (k+1) * N]; // load a right
    //             __m256d b = _mm256_load_pd((double*)&B[k + j*N]); // load b
    //             __m256d b_flip = _mm256_permute_pd(b, 0b0101); // flip within lane
    //             __m256d rr_ri = _mm256_mul_pd(_mm256_set1_pd(a_left.real()), b);
    //             __m256d ii_ir = _mm256_mul_pd(_mm256_set1_pd(a_left.imag()), b_flip);
    //             __m256d rr_minus_ii = _mm256_sub_pd(rr_ri, ii_ir);
    //             __m256d ri_plus_ir = _mm256_add_pd(rr_ri, ii_ir);
    //             __m256d a_left_times_b = _mm256_blend_pd(rr_minus_ii, ri_plus_ir, 0b1010);
    //             rr_ri = _mm256_mul_pd(_mm256_set1_pd(a_right.real()), b);
    //             ii_ir = _mm256_mul_pd(_mm256_set1_pd(a_right.imag()), b_flip);
    //             rr_minus_ii = _mm256_sub_pd(rr_ri, ii_ir);
    //             ri_plus_ir = _mm256_add_pd(rr_ri, ii_ir);
    //             __m256d a_right_times_b = _mm256_blend_pd(rr_minus_ii, ri_plus_ir, 0b1010);
    //             a_right_times_b = _mm256_permute2f128_pd(a_right_times_b, a_right_times_b, 0b01); // flip across lane
    //             c = _mm256_add_pd(c, _mm256_add_pd(a_left_times_b, a_right_times_b));
    //         }
    //         C[i + j * N] = floatType(c[0], c[1]);
    //     }
    // }

    // // Matrix multiplication using scalar * vector (AVX) WITH THREADING
    // // omp_set_num_threads(N * N); //TODO: Why this threading causes mem-out-of-bound error (segmentation fault)
    // omp_set_num_threads(omp_get_max_threads());
    // #pragma omp parallel
    // {
    // #pragma omp for collapse(2) schedule(dynamic) // jobs threaded across i j
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         __m256d c = _mm256_setzero_pd(); // c defined within loop avoiding race conditions
    //         for (int k = 0; k < N; k+=2) {
    //             floatType a_left = A[i + k * N]; // load a left
    //             floatType a_right = A[i + (k+1) * N]; // load a right
    //             __m256d b = _mm256_load_pd((double*)&B[k + j*N]); // load b
    //             __m256d b_flip = _mm256_permute_pd(b, 0b0101); // flip within lane
    //             __m256d rr_ri = _mm256_mul_pd(_mm256_set1_pd(a_left.real()), b);
    //             __m256d ii_ir = _mm256_mul_pd(_mm256_set1_pd(a_left.imag()), b_flip);
    //             __m256d rr_minus_ii = _mm256_sub_pd(rr_ri, ii_ir);
    //             __m256d ri_plus_ir = _mm256_add_pd(rr_ri, ii_ir);
    //             __m256d a_left_times_b = _mm256_blend_pd(rr_minus_ii, ri_plus_ir, 0b1010);
    //             rr_ri = _mm256_mul_pd(_mm256_set1_pd(a_right.real()), b);
    //             ii_ir = _mm256_mul_pd(_mm256_set1_pd(a_right.imag()), b_flip);
    //             rr_minus_ii = _mm256_sub_pd(rr_ri, ii_ir);
    //             ri_plus_ir = _mm256_add_pd(rr_ri, ii_ir);
    //             __m256d a_right_times_b = _mm256_blend_pd(rr_minus_ii, ri_plus_ir, 0b1010);
    //             a_right_times_b = _mm256_permute2f128_pd(a_right_times_b, a_right_times_b, 0b01); // flip across lane
    //             c = _mm256_add_pd(c, _mm256_add_pd(a_left_times_b, a_right_times_b));
    //         }
    //         C[i + j * N] = floatType(c[0], c[1]);
    //     }
    // }
    // }

    // // Matrix multiplication using vector * vector (AVX), NO THREADING
    // for (int i = 0; i < N; i+=2) {
    //     for (int j = 0; j < N; j+=2) {
    //         __m256d c_left = _mm256_setzero_pd(); __m256d c_right = _mm256_setzero_pd();
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
    //     }
    // }

    // Matrix multiplication using vector * vector (AVX) WITH THREADING
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel
    {
    #pragma omp for collapse(2) schedule(dynamic) // jobs threaded across i j
    for (int i = 0; i < N; i+=2) {
        for (int j = 0; j < N; j+=2) {
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

}