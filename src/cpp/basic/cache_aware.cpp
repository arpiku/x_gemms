#ifndef CACHE_AWARE_GEMM_H
#define CACHE_AWARE_GEMM_H

#include <cstddef>
#include <algorithm>

template <typename T>
void gemm_cache_aware(const T* A, const T* B, T* C, std::size_t N, std::size_t block_size = 32) {
    constexpr std::size_t L1_SIZE = 32 * 1024;
    constexpr std::size_t L2_SIZE = 256 * 1024;
    
    std::size_t l1_block = L1_SIZE / (3 * sizeof(T));
    std::size_t l2_block = L2_SIZE / (3 * sizeof(T));
    
    std::size_t b1 = std::min(block_size, l1_block);
    std::size_t b2 = std::min(block_size * 4, l2_block);

    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            C[i * N + j] = 0;
        }
    }

    for (std::size_t ii = 0; ii < N; ii += b2) {
        for (std::size_t jj = 0; jj < N; jj += b2) {
            for (std::size_t kk = 0; kk < N; kk += b2) {
                std::size_t i_max = std::min(ii + b2, N);
                std::size_t j_max = std::min(jj + b2, N);
                std::size_t k_max = std::min(kk + b2, N);

                for (std::size_t i = ii; i < i_max; i += b1) {
                    std::size_t i_b = std::min(i + b1, i_max);
                    for (std::size_t k = kk; k < k_max; k += b1) {
                        std::size_t k_b = std::min(k + b1, k_max);
                        for (std::size_t j = jj; j < j_max; j += b1) {
                            std::size_t j_b = std::min(j + b1, j_max);
                            for (std::size_t ii = i; ii < i_b; ++ii) {
                                for (std::size_t kk = k; kk < k_b; ++kk) {
                                    T a = A[ii * N + kk];
                                    for (std::size_t jj = j; jj < j_b; ++jj) {
                                        C[ii * N + jj] += a * B[kk * N + jj];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template void gemm_cache_aware<float>(const float*, const float*, float*, std::size_t, std::size_t);
template void gemm_cache_aware<double>(const double*, const double*, double*, std::size_t, std::size_t);

#endif
