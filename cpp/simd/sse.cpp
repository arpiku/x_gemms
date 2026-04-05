#ifndef SSE_GEMM_H
#define SSE_GEMM_H

#include <cstddef>
#include <emmintrin.h>

void gemm_sse(const float* A, const float* B, float* C, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; j += 4) {
            __m128 c_vals = _mm_setzero_ps();
            for (std::size_t k = 0; k < N; ++k) {
                __m128 a_val = _mm_set1_ps(A[i * N + k]);
                __m128 b_vals = _mm_loadu_ps(&B[k * N + j]);
                c_vals = _mm_add_ps(c_vals, _mm_mul_ps(a_val, b_vals));
            }
            _mm_storeu_ps(&C[i * N + j], c_vals);
        }
    }
}

#endif
