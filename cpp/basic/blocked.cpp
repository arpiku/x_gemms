#ifndef BLOCKED_GEMM_H
#define BLOCKED_GEMM_H

#include <cstddef>
#include <algorithm>

template <typename T>
void gemm_blocked(const T* A, const T* B, T* C, std::size_t N, std::size_t block_size = 64) {
    for (std::size_t ii = 0; ii < N; ii += block_size) {
        for (std::size_t jj = 0; jj < N; jj += block_size) {
            for (std::size_t kk = 0; kk < N; kk += block_size) {
                std::size_t i_max = std::min(ii + block_size, N);
                std::size_t j_max = std::min(jj + block_size, N);
                std::size_t k_max = std::min(kk + block_size, N);

                for (std::size_t i = ii; i < i_max; ++i) {
                    for (std::size_t k = kk; k < k_max; ++k) {
                        T a_ik = A[i * N + k];
                        for (std::size_t j = jj; j < j_max; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

template void gemm_blocked<float>(const float*, const float*, float*, std::size_t, std::size_t);
template void gemm_blocked<double>(const double*, const double*, double*, std::size_t, std::size_t);

#endif
