#ifndef AVX512_GEMM_H
#define AVX512_GEMM_H

#include <cstddef>
#include <algorithm>
#include <immintrin.h>

void gemm_avx512(const float* A, const float* B, float* C, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; j += 16) {
            __m512 c_vals = _mm512_setzero_ps();
            for (std::size_t k = 0; k < N; ++k) {
                __m512 a_val = _mm512_set1_ps(A[i * N + k]);
                __m512 b_vals = _mm512_loadu_ps(&B[k * N + j]);
                c_vals = _mm512_add_ps(c_vals, _mm512_mul_ps(a_val, b_vals));
            }
            _mm512_storeu_ps(&C[i * N + j], c_vals);
        }
    }
}

void gemm_avx512_blocked(const float* A, const float* B, float* C, std::size_t N, std::size_t block = 64) {
    for (std::size_t ii = 0; ii < N; ii += block) {
        for (std::size_t jj = 0; jj < N; jj += block) {
            for (std::size_t kk = 0; kk < N; kk += block) {
                std::size_t i_max = std::min(ii + block, N);
                std::size_t j_max = std::min(jj + block, N);
                std::size_t k_max = std::min(kk + block, N);

                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        __m512 a_val = _mm512_set1_ps(A[i * N + k]);
                        std::size_t j = jj;
                        for (; j + 16 <= j_max; j += 16) {
                            __m512 b_vals = _mm512_loadu_ps(&B[k * N + j]);
                            __m512 c_vals = _mm512_loadu_ps(&C[i * N + j]);
                            c_vals = _mm512_add_ps(c_vals, _mm512_mul_ps(a_val, b_vals));
                            _mm512_storeu_ps(&C[i * N + j], c_vals);
                        }
                        for (; j < j_max; ++j) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

#endif
