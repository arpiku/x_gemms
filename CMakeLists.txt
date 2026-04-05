cmake_minimum_required(VERSION 3.18)
project(x_gemms CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP REQUIRED)

add_executable(gemm_bench cpp/main.cpp)
target_include_directories(gemm_bench PRIVATE cpp)
target_link_libraries(gemm_bench PRIVATE OpenMP::OpenMP_CXX)

enable_language(CUDA)

set(CUDA_NVCC_FLAGS "-O3 --use_fast_math")
set(CUDA_ARCH "80")

add_executable(gpu_bench gpu/main.cu)
target_link_libraries(gpu_bench PRIVATE cublas cudart)
