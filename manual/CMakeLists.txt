cmake_minimum_required(VERSION 3.18)
project(matmul CUDA CXX)

# Set C++ standard to C++20 (required by ThunderKittens)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 80 86)

# Add ThunderKittens include directory
include_directories(${CMAKE_SOURCE_DIR}/ThunderKittens/include)

# Add executable
add_executable(matmul thunderkitten_matmul.cu)

# Set CUDA specific flags
target_compile_options(matmul PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    -O3
    --use_fast_math
    -lineinfo
    --extended-lambda
    --expt-relaxed-constexpr
    --threads 0
    --ptxas-options=-v
    -arch=sm_80
    -DTHUNDERKITTENS_ENABLE_TENSOR_CORES
>) 