#ifndef STRASSEN_COMMON_H
#define STRASSEN_COMMON_H

#include <cstddef>

template <typename T>
inline void strassen_add_matrix(const T* A, const T* B, T* C, size_t N) {
    for (size_t i = 0; i < N * N; ++i) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
inline void strassen_sub_matrix(const T* A, const T* B, T* C, size_t N) {
    for (size_t i = 0; i < N * N; ++i) {
        C[i] = A[i] - B[i];
    }
}

#endif
