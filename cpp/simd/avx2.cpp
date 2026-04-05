#ifndef AVX2_GEMM_H
#define AVX2_GEMM_H

#include <cstddef>
#include <algorithm>
#include <immintrin.h>

void gemm_avx2(const float* A, const float* B, float* C, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; j += 8) {
            __m256 c_vals = _mm256_setzero_ps();
            for (std::size_t k = 0; k < N; ++k) {
                __m256 a_val = _mm256_set1_ps(A[i * N + k]);
                __m256 b_vals = _mm256_loadu_ps(&B[k * N + j]);
                c_vals = _mm256_add_ps(c_vals, _mm256_mul_ps(a_val, b_vals));
            }
            _mm256_storeu_ps(&C[i * N + j], c_vals);
        }
    }
}

void gemm_avx2_blocked(const float* A, const float* B, float* C, std::size_t N, std::size_t block = 64) {
    for (std::size_t ii = 0; ii < N; ii += block) {
        for (std::size_t jj = 0; jj < N; jj += block) {
            for (std::size_t kk = 0; kk < N; kk += block) {
                std::size_t i_max = std::min(ii + block, N);
                std::size_t j_max = std::min(jj + block, N);
                std::size_t k_max = std::min(kk + block, N);

                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        __m256 a_val = _mm256_set1_ps(A[i * N + k]);
                        std::size_t j = jj;
                        for (; j + 8 <= j_max; j += 8) {
                            __m256 b_vals = _mm256_loadu_ps(&B[k * N + j]);
                            __m256 c_vals = _mm256_loadu_ps(&C[i * N + j]);
                            c_vals = _mm256_add_ps(c_vals, _mm256_mul_ps(a_val, b_vals));
                            _mm256_storeu_ps(&C[i * N + j], c_vals);
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
