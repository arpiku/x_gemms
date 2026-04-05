#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include "basic/naive.cpp"
#include "basic/blocked.cpp"
#include "basic/cache_aware.cpp"
#include "simd/sse.cpp"
#include "simd/avx2.cpp"
#ifdef __AVX512F__
#include "simd/avx512.cpp"
#endif
#include "parallel/openmp.cpp"
#include "parallel/std_thread.cpp"
#include "strassen/naive.cpp"
#include "strassen/blocked.cpp"
#include "strassen/cache_aware.cpp"

template <typename T>
void initialize_matrix(T* A, size_t N) {
    #pragma omp parallel for
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = static_cast<T>(rand() % 100) / 10.0f;
    }
}

template <typename T>
double benchmark_naive(T* A, T* B, T* C, size_t N, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_naive(A, B, C, N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_naive(A, B, C, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_blocked(T* A, T* B, T* C, size_t N, size_t block_size = 64, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_blocked(A, B, C, N, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_blocked(A, B, C, N, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_cache_aware(T* A, T* B, T* C, size_t N, size_t block_size = 32, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_cache_aware(A, B, C, N, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_cache_aware(A, B, C, N, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

double benchmark_sse(float* A, float* B, float* C, size_t N, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(float));
        gemm_sse(A, B, C, N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(float));
        gemm_sse(A, B, C, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

double benchmark_avx2(float* A, float* B, float* C, size_t N, size_t block_size = 64, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(float));
        gemm_avx2_blocked(A, B, C, N, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(float));
        gemm_avx2_blocked(A, B, C, N, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

double benchmark_avx512(float* A, float* B, float* C, size_t N, size_t block_size = 64, int iterations = 10) {
#ifdef __AVX512F__
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(float));
        gemm_avx512_blocked(A, B, C, N, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(float));
        gemm_avx512_blocked(A, B, C, N, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
#else
    return benchmark_avx2(A, B, C, N, block_size, iterations);
#endif
}

template <typename T>
double benchmark_openmp(T* A, T* B, T* C, size_t N, int num_threads = 0, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_openmp(A, B, C, N, num_threads);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_openmp(A, B, C, N, num_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_openmp_blocked(T* A, T* B, T* C, size_t N, size_t block_size = 64, int num_threads = 0, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_openmp_blocked(A, B, C, N, block_size, num_threads);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_openmp_blocked(A, B, C, N, block_size, num_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_std_thread(T* A, T* B, T* C, size_t N, int num_threads = 0, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_std_thread(A, B, C, N, num_threads);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_std_thread(A, B, C, N, num_threads);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_strassen_naive(T* A, T* B, T* C, size_t N, size_t threshold = 64, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_strassen_naive(A, B, C, N, threshold);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_strassen_naive(A, B, C, N, threshold);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_strassen_blocked(T* A, T* B, T* C, size_t N, size_t threshold = 64, size_t block_size = 64, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_strassen_blocked(A, B, C, N, threshold, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_strassen_blocked(A, B, C, N, threshold, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_strassen_cache_aware(T* A, T* B, T* C, size_t N, size_t threshold = 64, size_t block_size = 32, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_strassen_cache_aware(A, B, C, N, threshold, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        gemm_strassen_cache_aware(A, B, C, N, threshold, block_size);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

void print_results(const char* name, size_t N, double time_ms) {
    double gflops = (2.0 * N * N * N) / (time_ms * 1e6);
    double bytes = 3.0 * N * N * 4.0;
    double bandwidth = bytes / (time_ms * 1e6);
    std::cout << name << "," << N << "," << time_ms << "," << gflops << "," << bandwidth << std::endl;
}

int main(int argc, char* argv[]) {
    // Default sizes (MEDIUM_SIZES from config)
    std::vector<size_t> default_sizes = {64, 128, 256, 512, 1024, 2048};
    
    // Parse arguments
    // Usage: ./gemm_bench <num_threads> [sizes...]
    // Example: ./gemm_bench 20 64 128 256 512 1024 2048
    
    int num_threads = 0;
    std::vector<size_t> sizes;
    bool sizes_provided = false;
    
    // First arg is num_threads if it's a number
    int arg_idx = 1;
    if (argc > 1) {
        // Check if first arg is a number (thread count)
        char* endptr;
        long threads = strtol(argv[1], &endptr, 10);
        if (endptr != argv[1] && *endptr == '\0') {
            num_threads = (int)threads;
            arg_idx = 2;
        }
    }
    
    // Remaining args are sizes
    for (int i = arg_idx; i < argc; i++) {
        char* endptr;
        long size = strtol(argv[i], &endptr, 10);
        if (endptr != argv[i] && *endptr == '\0' && size > 0) {
            sizes.push_back((size_t)size);
            sizes_provided = true;
        }
    }
    
    // If no sizes provided, use default and note it
    if (!sizes_provided) {
        sizes = default_sizes;
        std::cerr << "INFO: No sizes provided, using default MEDIUM_SIZES (64,128,256,512,1024,2048)" << std::endl;
    }
    
    std::cout << "algorithm,size,time_ms,gfops,bandwidth_gbs" << std::endl;

    // OPTIMIZATION: Generate matrices once for MAX size, use subsets for smaller sizes
    // This reduces initialization time significantly (~6x for medium tests)
    size_t max_size = sizes[0];
    for (size_t s : sizes) {
        if (s > max_size) max_size = s;
    }
    
    float *A = (float*)malloc(max_size * max_size * sizeof(float));
    float *B = (float*)malloc(max_size * max_size * sizeof(float));
    float *C = (float*)malloc(max_size * max_size * sizeof(float));
    
    initialize_matrix(A, max_size);
    initialize_matrix(B, max_size);

    for (size_t N : sizes) {
        // Matrices are already allocated for max_size
        // Algorithms only access A[0:N][0:N], B[0:N][0:N] based on N parameter

        double time = benchmark_naive(A, B, C, N);
        print_results("naive", N, time);

        time = benchmark_blocked(A, B, C, N, 64);
        print_results("blocked", N, time);

        time = benchmark_cache_aware(A, B, C, N, 32);
        print_results("cache_aware", N, time);

        time = benchmark_sse(A, B, C, N);
        print_results("sse", N, time);

        time = benchmark_avx2(A, B, C, N, 64);
        print_results("avx2", N, time);

#ifdef __AVX512F__
        time = benchmark_avx512(A, B, C, N, 64);
        print_results("avx512", N, time);
#endif

        time = benchmark_openmp(A, B, C, N, num_threads);
        print_results("openmp", N, time);

        time = benchmark_openmp_blocked(A, B, C, N, 64, num_threads);
        print_results("openmp_blocked", N, time);

        time = benchmark_std_thread(A, B, C, N, num_threads);
        print_results("std_thread", N, time);

        time = benchmark_strassen_naive(A, B, C, N, 64);
        print_results("strassen_naive", N, time);

        time = benchmark_strassen_blocked(A, B, C, N, 64, 64);
        print_results("strassen_blocked", N, time);

        time = benchmark_strassen_cache_aware(A, B, C, N, 64, 32);
        print_results("strassen_cache_aware", N, time);
    }

    // Free matrices after all sizes are processed
    free(A);
    free(B);
    free(C);

    return 0;
}
