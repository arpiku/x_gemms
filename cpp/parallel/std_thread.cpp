#ifndef STD_THREAD_GEMM_H
#define STD_THREAD_GEMM_H

#include <cstddef>
#include <algorithm>
#include <thread>
#include <vector>

template <typename T>
void gemm_std_thread(const T* A, const T* B, T* C, std::size_t N, int num_threads = 0) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    auto worker = [&](std::size_t start_row, std::size_t end_row) {
        for (std::size_t i = start_row; i < end_row; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                T sum = 0;
                for (std::size_t k = 0; k < N; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    };

    std::vector<std::thread> threads;
    std::size_t rows_per_thread = N / num_threads;
    
    for (int t = 0; t < num_threads - 1; ++t) {
        threads.emplace_back(worker, t * rows_per_thread, (t + 1) * rows_per_thread);
    }
    threads.emplace_back(worker, (num_threads - 1) * rows_per_thread, N);

    for (auto& t : threads) {
        t.join();
    }
}

template <typename T>
void gemm_std_thread_blocked(const T* A, const T* B, T* C, std::size_t N, std::size_t block_size = 64, int num_threads = 0) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    auto worker = [&](std::size_t start_block, std::size_t num_blocks) {
        for (std::size_t b = start_block; b < start_block + num_blocks; ++b) {
            std::size_t ii = (b * block_size) % N;
            std::size_t jj = (b * block_size) / (N / block_size) * block_size;
            jj = jj % N;

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
    };

    std::vector<std::thread> threads;
    std::size_t total_blocks = (N / block_size) * (N / block_size);
    std::size_t blocks_per_thread = total_blocks / num_threads;

    for (int t = 0; t < num_threads - 1; ++t) {
        threads.emplace_back(worker, t * blocks_per_thread, blocks_per_thread);
    }
    threads.emplace_back(worker, (num_threads - 1) * blocks_per_thread, 
                        total_blocks - (num_threads - 1) * blocks_per_thread);

    for (auto& t : threads) {
        t.join();
    }
}

template void gemm_std_thread<float>(const float*, const float*, float*, std::size_t, int);
template void gemm_std_thread<double>(const double*, const double*, double*, std::size_t, int);
template void gemm_std_thread_blocked<float>(const float*, const float*, float*, std::size_t, std::size_t, int);
template void gemm_std_thread_blocked<double>(const double*, const double*, double*, std::size_t, std::size_t, int);

#endif
