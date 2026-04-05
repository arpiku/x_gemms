#include <iostream>
#include <chrono>
#include <cstring>
#include <cmath>
#include "basic/naive.cpp"
#include "basic/blocked.cpp"
#include "basic/cache_aware.cpp"
#include "simd/sse.cpp"
#include "simd/avx2.cpp"
#include "simd/avx512.cpp"
#include "parallel/openmp.cpp"
#include "parallel/std_thread.cpp"

template <typename T>
void initialize_matrix(T* A, size_t N) {
    for (size_t i = 0; i < N * N; ++i) {
        A[i] = static_cast<T>(rand() % 100) / 10.0f;
    }
}

template <typename T>
double benchmark_function(void (*func)(const T*, const T*, T*, size_t), T* A, T* B, T* C, size_t N, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        func(A, B, C, N);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        func(A, B, C, N);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

template <typename T>
double benchmark_function_blocked(void (*func)(const T*, const T*, T*, size_t, size_t), T* A, T* B, T* C, size_t N, size_t block_size = 64, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        std::memset(C, 0, N * N * sizeof(T));
        func(A, B, C, N, block_size);
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::memset(C, 0, N * N * sizeof(T));
        func(A, B, C, N, block_size);
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
    size_t sizes[] = {64, 128, 256, 512, 1024};
    int num_threads = argc > 1 ? std::atoi(argv[1]) : 0;

    std::cout << "algorithm,size,time_ms,gfops,bandwidth_gbs" << std::endl;

    for (size_t N : sizes) {
        float *A = (float*)malloc(N * N * sizeof(float));
        float *B = (float*)malloc(N * N * sizeof(float));
        float *C = (float*)malloc(N * N * sizeof(float));

        initialize_matrix(A, N);
        initialize_matrix(B, N);

        double time = benchmark_function(gemm_naive<float>, A, B, C, N);
        print_results("naive", N, time);

        time = benchmark_function_blocked(gemm_blocked<float>, A, B, C, N, 64);
        print_results("blocked", N, time);

        time = benchmark_function_blocked(gemm_cache_aware<float>, A, B, C, N, 32);
        print_results("cache_aware", N, time);

        time = benchmark_function(gemm_sse, A, B, C, N);
        print_results("sse", N, time);

        time = benchmark_function_blocked(gemm_avx2_blocked, A, B, C, N, 64);
        print_results("avx2", N, time);

        time = benchmark_function_blocked(gemm_avx512_blocked, A, B, C, N, 64);
        print_results("avx512", N, time);

        time = benchmark_function(gemm_openmp<float>, A, B, C, N, num_threads);
        print_results("openmp", N, time);

        time = benchmark_function_blocked(gemm_openmp_blocked<float>, A, B, C, N, 64, num_threads);
        print_results("openmp_blocked", N, time);

        time = benchmark_function(gemm_std_thread<float>, A, B, C, N, num_threads);
        print_results("std_thread", N, time);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
