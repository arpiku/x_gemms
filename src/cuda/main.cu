#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstdlib>

__global__ void gemm_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void gemm_cuda_naive(const float* A, const float* B, float* C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    gemm_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void gemm_cublas(const float* A, const float* B, float* C, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);
    float alpha = 1.0f, beta = 0.0f;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
}

void initialize_matrix(float* A, int N) {
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand() % 100) / 10.0f;
    }
}

double benchmark_function(void (*func)(const float*, const float*, float*, int), float* A, float* B, float* C, int N, int iterations = 10) {
    for (int w = 0; w < 3; ++w) {
        func(A, B, C, N);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; ++i) {
        func(A, B, C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / iterations;
}

void print_results(const char* name, int N, double time_ms) {
    double gflops = (2.0 * N * N * N) / (time_ms * 1e6);
    double bytes = 3.0 * N * N * 4.0;
    double bandwidth = bytes / (time_ms * 1e6);
    printf("%s,%d,%.2f,%.2f,%.2f\n", name, N, time_ms, gflops, bandwidth);
}

int main(int argc, char* argv[]) {
    // Default sizes (MEDIUM_SIZES from config)
    std::vector<int> default_sizes = {64, 128, 256, 512, 1024, 2048};
    
    // Parse arguments
    // Usage: ./gpu_bench [sizes...]
    // Example: ./gpu_bench 64 128 256 512 1024 2048
    
    std::vector<int> sizes;
    bool sizes_provided = false;
    
    for (int i = 1; i < argc; i++) {
        char* endptr;
        long size = strtol(argv[i], &endptr, 10);
        if (endptr != argv[i] && *endptr == '\0' && size > 0) {
            sizes.push_back((int)size);
            sizes_provided = true;
        }
    }
    
    // If no sizes provided, use default
    if (!sizes_provided) {
        sizes = default_sizes;
        fprintf(stderr, "INFO: No sizes provided, using default MEDIUM_SIZES (64,128,256,512,1024,2048)\n");
    }
    
    printf("algorithm,size,time_ms,gfops,bandwidth_gbs\n");

    for (int N : sizes) {
        float *A = (float*)malloc(N * N * sizeof(float));
        float *B = (float*)malloc(N * N * sizeof(float));
        float *C = (float*)malloc(N * N * sizeof(float));

        initialize_matrix(A, N);
        initialize_matrix(B, N);

        double time = benchmark_function(gemm_cuda_naive, A, B, C, N);
        print_results("cuda_naive", N, time);

        time = benchmark_function(gemm_cublas, A, B, C, N);
        print_results("cublas", N, time);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
