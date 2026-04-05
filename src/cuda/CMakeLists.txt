cmake_minimum_required(VERSION 3.18)
project(gpu_bench CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# RTX 5070 Blackwell = sm_120
set(CUDA_NVCC_FLAGS "-O3 --use_fast_math -arch=sm_120")

find_package(CUDAToolkit REQUIRED)

add_executable(gpu_bench main.cu)
target_link_libraries(gpu_bench PRIVATE CUDA::cuda_runtime CUDA::cublas)
