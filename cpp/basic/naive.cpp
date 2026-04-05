#ifndef NAIVE_GEMM_H
#define NAIVE_GEMM_H

#include <cstddef>
#include <cstring>

template <typename T>
void gemm_naive(const T* A, const T* B, T* C, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            T sum = 0;
            for (std::size_t k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

template void gemm_naive<float>(const float*, const float*, float*, std::size_t);
template void gemm_naive<double>(const double*, const double*, double*, std::size_t);

#endif
