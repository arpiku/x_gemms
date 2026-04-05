#ifndef STRASSEN_CACHE_AWARE_H
#define STRASSEN_CACHE_AWARE_H

#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include "common.h"

constexpr size_t STRASSEN_THRESHOLD_CACHE = 64;
constexpr size_t CACHE_BLOCK_SIZE = 32;

template <typename T>
class StrassenMemoryPool {
private:
    T* pool;
    size_t capacity;
    size_t offset;
    
public:
    StrassenMemoryPool(size_t max_n) {
        size_t total_elements = 0;
        size_t n = max_n;
        while (n > STRASSEN_THRESHOLD_CACHE && n % 2 == 0) {
            size_t sub_n = n / 2;
            size_t sub_size = sub_n * sub_n;
            total_elements += 8 * sub_size + 7 * sub_size + 4 * sub_size + 2 * sub_size;
            n = sub_n;
        }
        capacity = total_elements;
        pool = (T*)malloc(capacity * sizeof(T));
        offset = 0;
    }
    
    ~StrassenMemoryPool() {
        free(pool);
    }
    
    T* allocate(size_t count) {
        T* ptr = pool + offset;
        offset += count;
        return ptr;
    }
    
    void reset() {
        offset = 0;
    }
};

template <typename T>
void gemm_cache_aware_base(const T* A, const T* B, T* C, size_t N, size_t block_size = CACHE_BLOCK_SIZE) {
    for (size_t ii = 0; ii < N; ii += block_size) {
        for (size_t jj = 0; jj < N; jj += block_size) {
            for (size_t kk = 0; kk < N; kk += block_size) {
                size_t i_max = std::min(ii + block_size, N);
                size_t j_max = std::min(jj + block_size, N);
                size_t k_max = std::min(kk + block_size, N);
                
                for (size_t i = ii; i < i_max; ++i) {
                    for (size_t k = kk; k < k_max; ++k) {
                        T a_ik = A[i * N + k];
                        for (size_t j = jj; j < j_max; ++j) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void strassen_cache_aware_impl(const T* A, const T* B, T* C, size_t N, 
                                size_t threshold, size_t block_size,
                                StrassenMemoryPool<T>& pool) {
    if (N <= threshold) {
        gemm_cache_aware_base(A, B, C, N, block_size);
        return;
    }
    
    if (N % 2 != 0) {
        gemm_cache_aware_base(A, B, C, N, block_size);
        return;
    }
    
    size_t mid = N / 2;
    size_t sub_size = mid * mid;
    
    size_t saved_offset = pool.allocate(0) - pool.allocate(0);
    
    T* A11 = pool.allocate(sub_size);
    T* A12 = pool.allocate(sub_size);
    T* A21 = pool.allocate(sub_size);
    T* A22 = pool.allocate(sub_size);
    T* B11 = pool.allocate(sub_size);
    T* B12 = pool.allocate(sub_size);
    T* B21 = pool.allocate(sub_size);
    T* B22 = pool.allocate(sub_size);
    
    for (size_t i = 0; i < mid; ++i) {
        for (size_t j = 0; j < mid; ++j) {
            A11[i * mid + j] = A[i * N + j];
            A12[i * mid + j] = A[i * N + mid + j];
            A21[i * mid + j] = A[(mid + i) * N + j];
            A22[i * mid + j] = A[(mid + i) * N + mid + j];
            
            B11[i * mid + j] = B[i * N + j];
            B12[i * mid + j] = B[i * N + mid + j];
            B21[i * mid + j] = B[(mid + i) * N + j];
            B22[i * mid + j] = B[(mid + i) * N + mid + j];
        }
    }
    
    T* T1 = pool.allocate(sub_size);
    T* T2 = pool.allocate(sub_size);
    
    T* M1 = pool.allocate(sub_size);
    T* M2 = pool.allocate(sub_size);
    T* M3 = pool.allocate(sub_size);
    T* M4 = pool.allocate(sub_size);
    T* M5 = pool.allocate(sub_size);
    T* M6 = pool.allocate(sub_size);
    T* M7 = pool.allocate(sub_size);
    
    T* C11 = pool.allocate(sub_size);
    T* C12 = pool.allocate(sub_size);
    T* C21 = pool.allocate(sub_size);
    T* C22 = pool.allocate(sub_size);
    
    strassen_add_matrix(A11, A22, T1, mid);
    strassen_add_matrix(B11, B22, T2, mid);
    strassen_cache_aware_impl(T1, T2, M1, mid, threshold, block_size, pool);
    
    strassen_add_matrix(A21, A22, T1, mid);
    strassen_cache_aware_impl(T1, B11, M2, mid, threshold, block_size, pool);
    
    strassen_sub_matrix(B12, B22, T1, mid);
    strassen_cache_aware_impl(A11, T1, M3, mid, threshold, block_size, pool);
    
    strassen_sub_matrix(B21, B11, T1, mid);
    strassen_cache_aware_impl(A22, T1, M4, mid, threshold, block_size, pool);
    
    strassen_add_matrix(A11, A12, T1, mid);
    strassen_cache_aware_impl(T1, B22, M5, mid, threshold, block_size, pool);
    
    strassen_sub_matrix(A21, A11, T1, mid);
    strassen_add_matrix(B11, B12, T2, mid);
    strassen_cache_aware_impl(T1, T2, M6, mid, threshold, block_size, pool);
    
    strassen_sub_matrix(A12, A22, T1, mid);
    strassen_add_matrix(B21, B22, T2, mid);
    strassen_cache_aware_impl(T1, T2, M7, mid, threshold, block_size, pool);
    
    strassen_add_matrix(M1, M4, C11, mid);
    strassen_sub_matrix(C11, M5, C11, mid);
    strassen_add_matrix(C11, M7, C11, mid);
    
    strassen_add_matrix(M3, M5, C12, mid);
    
    strassen_add_matrix(M2, M4, C21, mid);
    
    strassen_sub_matrix(M1, M2, C22, mid);
    strassen_add_matrix(C22, M3, C22, mid);
    strassen_add_matrix(C22, M6, C22, mid);
    
    for (size_t i = 0; i < mid; ++i) {
        for (size_t j = 0; j < mid; ++j) {
            C[i * N + j] = C11[i * mid + j];
            C[i * N + mid + j] = C12[i * mid + j];
            C[(mid + i) * N + j] = C21[i * mid + j];
            C[(mid + i) * N + mid + j] = C22[i * mid + j];
        }
    }
}

template <typename T>
void gemm_strassen_cache_aware(const T* A, const T* B, T* C, size_t N, 
                                size_t threshold = STRASSEN_THRESHOLD_CACHE, 
                                size_t block_size = CACHE_BLOCK_SIZE) {
    std::memset(C, 0, N * N * sizeof(T));
    StrassenMemoryPool<T> pool(N);
    strassen_cache_aware_impl(A, B, C, N, threshold, block_size, pool);
}

template void gemm_strassen_cache_aware<float>(const float*, const float*, float*, size_t, size_t, size_t);
template void gemm_strassen_cache_aware<double>(const double*, const double*, double*, size_t, size_t, size_t);

#endif
